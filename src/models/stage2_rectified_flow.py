# -*- coding: UTF-8 -*-
"""
@Time : 16/06/2025 18:03
@Author : Xiaoguang Liang
@File : stage2_rectified_flow.py
@Project : Enhancing_Sketch-to-3D_Controllability
"""
import random
from typing import List

from tqdm import trange, tqdm
import torch
import torch.nn.functional as F
from PIL import Image
from torchvision import transforms
import numpy as np
from torch.cuda.amp import autocast, GradScaler

from src.models.stage1_rectified_flow import Stage1Model
from src.utils.image_util import merge_images, draw_text
from src.utils.visual_util import render_gaussians, render_mesh
from src.utils.common import np2th
from src.utils.spaghetti_util import (generate_zc_from_sj_gaus,
                                      get_mesh_from_spaghetti, load_marching_cube_meshing,
                                      load_spaghetti)
from src.utils.train_util import get_dropout_mask, load_model
from src.utils.image_util import build_transforms
from configs.global_setting import DATA_DIR, device
from configs.log_config import logger
from src.dataset_preparation.spaghetti_dataset import DatasetBuilder
from src.utils.data_util import get_test_images
from src.utils.train_util import compute_loss_weighting_for_sd3, compute_density_for_timestep_sampling
from src.model_components.utils.misc import instantiate_from_config
from src.inference.pipelines import retrieve_timesteps
from src.model_components.utils.ema import LitEma
from src.inference.pipeline_for_spaghetti import RectifiedFlowPipeline
from src.models import init_model


class Stage2Model(Stage1Model):
    def __init__(self, network, variance_schedule, train_cfg, scheduler_cfg, optimizer_cfg,
                 ema_config=None,
                 use_amp: bool = False,
                 torch_compile=False,
                 classifier_free_guidance=False,
                 free_guidance_weight=1.0,
                 num_inference_steps=50,
                 grad_clip: float = 1.0,
                 **kwargs):
        super().__init__(network, variance_schedule, train_cfg, scheduler_cfg, optimizer_cfg,
                         **kwargs)

        self.classifier_free_guidance = classifier_free_guidance
        self.free_guidance_weight = free_guidance_weight
        self.num_inference_steps = num_inference_steps

        self.transform_train, self.transform_test = build_transforms(self.hparams.dataset_kwargs)

        # ========= init image encoder config ========= #
        encoder_model = kwargs.get('encoder_model')
        self.image_encoder = init_model(encoder_model)

        # ========= config ema model ========= #
        self.ema_config = ema_config
        if self.ema_config is not None:
            if self.ema_config['ema_model'] == 'DSEma':
                # from michelangelo.models.modules.ema_deepspeed import DSEma
                from src.model_components.utils import DSEma
                self.model_ema = DSEma(self.net, decay=self.ema_config['ema_decay'])
            else:
                self.model_ema = LitEma(self.net, decay=self.ema_config['ema_decay'])
            self.model_ema.to(device)
            # do not initilize EMA weight from ckpt path, since I need to change moe layers
            print(f"Keeping EMAs of {len(list(self.model_ema.buffers()))}.")

        # ========= config stage 1 model for inference ========= #
        if not self.hparams.no_run_validation:
            # Initialize mesher
            self.mc_mesher = load_marching_cube_meshing(device=device)
            # Load stage1 model
            time_folder = self.hparams.stage1_model_path
            model_name = "stage1_rectified_flow"
            self.stage1_model = load_model(time_folder, Stage1Model, model_name, device)
            self.stage1_model.ema_config = None

        val_noise_schedule = instantiate_from_config(scheduler_cfg)
        self.pipeline = RectifiedFlowPipeline(self.net, val_noise_schedule, self.hparams)

        self.use_amp = use_amp
        self.scaler = GradScaler() if use_amp else None
        self.optimizer_type = self.optimizer_cfg['optimizer_type']

        if torch_compile:
            torch.nn.Module.compile(self.net)
            torch.nn.Module.compile(self.stage1_model)
            print(f'*' * 100)
            print(f'Compile model for acceleration')
            print(f'*' * 100)

    def random_mask_gaus(self, gaus):
        if self.hparams.get("classifier_free_guidance"):
            B = gaus.shape[0]
            random_dp_mask = get_dropout_mask(
                B, self.hparams.conditioning_dropout_prob, self.device
            )
            gaus = gaus * random_dp_mask.unsqueeze(1).unsqueeze(2)

        return gaus

    def forward(self, x, gaus, img):
        """
        Input:
            x: [B,G,512]
            gaus: [B,G,16]
            text: list of [B]
        """
        B, G = x.shape[:2]
        # gaus = self.random_mask_gaus(gaus)
        img_encoded = self.image_encode(img, B)
        cond = self.cond_from_gaus_img(gaus, img_encoded)

        return self.get_loss(x, cond)

    def step(self, batch, stage):
        x, gaus, text = batch
        loss = self(x, gaus, text)

        self.log(f"{stage}/loss", loss, on_step=stage == "train", prog_bar=True, on_epoch=True, logger=True)
        return loss

    def get_loss(self, x0, cond, t=None, noisy_in=False, beta_in=None, e_rand_in=None):
        noise = torch.randn_like(x0)
        # For weighting schemes where we sample timesteps non-uniformly
        u = compute_density_for_timestep_sampling(
            weighting_scheme=self.train_cfg["weighting_scheme"],
            batch_size=self.train_cfg['batch_size'],
            logit_mean=self.train_cfg["logit_mean"],
            logit_std=self.train_cfg["logit_std"],
            mode_scale=self.train_cfg["mode_scale"],
        )
        indices = (u * self.scheduler.config.num_train_timesteps).long()
        timesteps = self.scheduler.timesteps[indices].to(device)  # [M, ]
        # Repeat the timesteps for each part

        sigmas = self.get_sigmas(timesteps, len(x0.shape), torch.bfloat16)
        latent_model_input = noisy_latents = (1. - sigmas) * x0 + sigmas * noise

        model_pred = self.net(latent_model_input, timesteps, cond)

        if self.train_cfg["training_objective"] == "x0":  # Section 5 of https://arxiv.org/abs/2206.00364
            model_pred = model_pred * (-sigmas) + noisy_latents  # predicted x_0
            target = x0
        elif self.train_cfg["training_objective"] == 'v':  # flow matching
            target = noise - x0
        elif self.train_cfg["training_objective"] == '-v':  # reverse flow matching
            # The training objective for TripoSG is the reverse of the flow matching objective.
            # It uses "different directions", i.e., the negative velocity.
            # This is probably a mistake in engineering, not very harmful.
            # In TripoSG's rectified flow scheduler, prev_sample = sample + (sigma - sigma_next) * model_output
            # See TripoSG's scheduler https://github.com/VAST-AI-Research/TripoSG/blob/main/triposg/schedulers/scheduling_rectified_flow.py#L296
            # While in diffusers's flow matching scheduler, prev_sample = sample + (sigma_next - sigma) * model_output
            # See https://github.com/huggingface/diffusers/blob/main/src/diffusers/schedulers/scheduling_flow_match_euler_discrete.py#L454
            target = x0 - noise
        else:
            raise ValueError(f"Unknown training objective [{self.train_cfg['training_objective']}]")

        # For these weighting schemes use a uniform timestep sampling, so post-weight the loss
        weighting = compute_loss_weighting_for_sd3(
            self.train_cfg["weighting_scheme"],
            sigmas
        )

        loss = weighting * F.mse_loss(model_pred.float(), target.float(), reduction="none")
        loss = loss.mean(dim=list(range(1, len(loss.shape)))).mean()

        return loss

    def cond_from_gaus_img(self, gaus, img_encoded):
        gaus = np2th(gaus).to(self.device)
        G = gaus.shape[1]
        img_encoded = np2th(img_encoded).to(self.device)
        assert gaus.ndim == 3
        if img_encoded.ndim == 2:
            img_encoded = img_encoded.unsqueeze(1)
        img_encoded = img_encoded.expand(-1, G, -1)
        return torch.cat([gaus, img_encoded], -1)

    def generate_null_cond(self, B, G, image_shape):
        null_img_encoded = self.generate_null_img_encoded(image_shape, B)

        gaus = torch.zeros(B, G, 16, dtype=torch.float, device=self.device)
        return self.cond_from_gaus_img(gaus, null_img_encoded)

    @torch.no_grad()
    def sample(
            self,
            num_samples_or_image,
            gaussian_latent,
            return_traj=False,
            return_cond=False,
            classifier_free_guidance=False,
            free_guidance_weight=0.7,
            num_inference_steps=50,
            sigmas: List[float] = None,
    ):

        global img_encoded
        if isinstance(num_samples_or_image, int):
            batch_size = num_samples_or_image
            dataset_builder = DatasetBuilder(self.hparams)
            ds = dataset_builder.val_dataloader()
            batch_gaus = []
            batch_img = []
            for i in range(batch_size):
                _, gaus, img = ds[i]
                batch_gaus.append(gaus)
                batch_img.append(img)

            batch_gaus = torch.stack(batch_gaus, 0)
            batch_img = torch.stack(batch_img, 0)
            image_shape = batch_img.shape
            img_encoded = self.image_encode(batch_img, batch_size)
            cond = self.cond_from_gaus_img(batch_gaus, img_encoded).to(self.device)

        elif isinstance(num_samples_or_image, list):
            img_encoded, images = self.read_encode_images(num_samples_or_image)
            batch_size = images.shape[0]
            cond = self.cond_from_gaus_img(gaussian_latent, img_encoded)
            cond.to(self.device)
            image_shape = images.shape

        logger.info(f"Start Stage2 sampling ...")
        num_tokens = 16
        num_channels_latents = 512

        latents = self.pipeline(img_encoded, image_shape,
                                num_inference_steps=num_inference_steps,
                                num_tokens=num_tokens,
                                num_channels_latents=num_channels_latents,
                                guidance_scale=free_guidance_weight,
                                stage=2,
                                cond=cond)
        return latents

    @torch.no_grad()
    def validation(self):
        # Load spaghetti
        if self.hparams.spaghetti_tag == 'chairs_large_6755':
            spaghetti_tag = 'chairs_large'
        else:
            spaghetti_tag = self.hparams.spaghetti_tag
        spaghetti = load_spaghetti(device=device, tag=spaghetti_tag)
        test_sketch_path = DATA_DIR / self.hparams.dataset_kwargs.test_sketch_path
        test_images = get_test_images(test_sketch_path)

        # Stage1 sampling
        extrinsics = self.stage1_model.sampling_gaussians(test_images,
                                                          classifier_free_guidance=self.classifier_free_guidance,
                                                          free_guidance_weight=self.free_guidance_weight,
                                                          num_inference_steps=self.num_inference_steps
                                                          )
        # stage2 sampling
        with self.ema_scope("Sample"):
            # with torch.amp.autocast(device_type='cuda'):
            intrinsics = self.sample(test_images, extrinsics,
                                     free_guidance_weight=0.7)

            zcs = generate_zc_from_sj_gaus(spaghetti, intrinsics, extrinsics)

            k = 256
            image_res = (k, k)

            image_transform = transforms.Compose([transforms.Resize(image_res), ])
            all_img = []
            for i, x in enumerate(zcs):
                gaus_img = render_gaussians(extrinsics[i], resolution=image_res)

                try:
                    generated_mesh = get_mesh_from_spaghetti(spaghetti, self.mc_mesher, x, res=k)
                except Exception as e:
                    logger.error(f"Mesh generation failed: {e}")
                    continue

                v, f = generated_mesh
                mesh_img = render_mesh(v, f, resolution=image_res)

                sketch_path = test_images[i]
                sketch_img = Image.open(DATA_DIR / sketch_path).convert('RGB')
                sketch_img = image_transform(sketch_img)

                img = merge_images([sketch_img, gaus_img, mesh_img])
                all_img.append(img)

            if all_img:
                try:
                    self.logger.log_image('Generated results', all_img)
                except Exception as e:
                    logger.error(f"Wandb Logging failed: {e}")
            else:
                logger.error(f"No rendered images for 3D generation to log!!!")
