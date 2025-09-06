# -*- coding: UTF-8 -*-
"""
@Time : 16/06/2025 18:03
@Author : Xiaoguang Liang
@File : stage1_rectified_flow.py
@Project : Enhancing_Sketch-to-3D_Controllability
"""
import random
from typing import List, Tuple
from contextlib import contextmanager

import torch
from torch import Tensor
import torch.nn.functional as F
from tqdm import trange, tqdm
from PIL import Image
from diffusers.utils.torch_utils import randn_tensor
from torchvision import transforms
import numpy as np
from torch.cuda.amp import autocast, GradScaler
from torch.optim import lr_scheduler

from src.models.base_model import BaseModel
from src.utils.image_util import merge_images, draw_text, svg_to_img, SVG, build_transforms
from src.utils.visual_util import render_gaussians
from src.utils.spaghetti_util import clip_eigenvalues, project_eigenvectors
from src.models import init_model
from configs.global_setting import DATA_DIR, device
from configs.log_config import logger
from src.dataset_preparation.spaghetti_dataset import DatasetBuilder
from src.utils.data_util import get_test_images
from src.utils.train_util import compute_loss_weighting_for_sd3, compute_density_for_timestep_sampling
from src.model_components.utils.misc import instantiate_from_config
from src.inference.pipelines import retrieve_timesteps
from src.model_components.utils.ema import LitEma
from src.inference.pipeline_for_spaghetti import RectifiedFlowPipeline


class Stage1Model(BaseModel):
    def __init__(self, network, variance_schedule, train_cfg, scheduler_cfg, optimizer_cfg,
                 ema_config=None,
                 use_amp: bool = False,
                 torch_compile=False,
                 classifier_free_guidance=False,
                 free_guidance_weight=1.0,
                 num_inference_steps=50,
                 grad_clip: float = 1.0,
                 **kwargs):
        super().__init__(network, variance_schedule, **kwargs)

        self.transform_train, self.transform_test = build_transforms(self.hparams.dataset_kwargs)

        # ========= init image encoder config ========= #
        encoder_model = kwargs.get('encoder_model')
        self.image_encoder = init_model(encoder_model)

        self.train_cfg = train_cfg
        self.grad_clip = grad_clip

        # ========= init optimizer config ========= #
        self.optimizer_cfg = optimizer_cfg

        # ========= init diffusion scheduler ========= #
        self.scheduler_cfg = scheduler_cfg
        self.scheduler = instantiate_from_config(scheduler_cfg)

        self.classifier_free_guidance = classifier_free_guidance
        self.free_guidance_weight = free_guidance_weight
        self.num_inference_steps = num_inference_steps

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

        val_noise_schedule = instantiate_from_config(scheduler_cfg)
        self.pipeline = RectifiedFlowPipeline(self.net, val_noise_schedule, self.hparams)

        self.use_amp = use_amp
        self.scaler = GradScaler() if use_amp else None
        self.optimizer_type = self.optimizer_cfg['optimizer_type']

        if torch_compile:
            torch.nn.Module.compile(self.net)
            print(f'*' * 100)
            print(f'Compile model for acceleration')
            print(f'*' * 100)

    def forward(self, x, img):
        """
        Input:
            x: [B,G,16]
            text: list of length [B]
        """
        B, G = x.shape[:2]
        img_encoded = self.image_encode(img, B)
        return self.get_loss(x, img_encoded)

    def step(self, batch, stage: str):
        x, cond = batch
        loss = self(x, cond)

        return loss

    def training_step(self, batch, batch_idx):
        x = batch

        if self.optimizer_type == 'muon':
            # Zero gradients
            for optimizer in self.optimizers():
                optimizer.zero_grad()

            # Forward pass with gradient accumulation
            if self.use_amp:
                # with torch.amp.autocast(device_type='cuda'):
                loss = self.step(x, "train")

                self.scaler.scale(loss).backward()
                # Unscale and clip gradients
                for optimizer in self.optimizers():
                    self.scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(self.net.parameters(), self.grad_clip)
                # Step optimizers
                for optimizer in self.optimizers():
                    self.scaler.step(optimizer)
                for scheduler in self.lr_schedulers():
                    scheduler.step()

                self.scaler.update()
            else:
                loss = self.step(x, "train")
                loss.backward()

                torch.nn.utils.clip_grad_norm_(self.net.parameters(), self.grad_clip)

                for optimizer in self.optimizers():
                    optimizer.step()
                for scheduler in self.lr_schedulers():
                    scheduler.step()

            lr_abs = self.optimizers()[0].param_groups[0]['lr']
        else:
            loss = self.step(x, "train")
            lr_abs = self.optimizers().param_groups[0]['lr']

        split = 'train'
        loss_dict = {
            f"{split}/simple": loss.detach(),
            f"{split}/total_loss": loss.detach(),
            f"{split}/lr_abs": lr_abs,
        }
        self.log_dict(loss_dict, prog_bar=True, logger=True, sync_dist=False, rank_zero_only=True)
        return loss

    def image_encode(self, images, batch_size):
        if self.hparams.get("image_encoder_return_seq"):
            img_encoded = self.image_encoder.encode(images).last_hidden_state
        else:
            img_encoded = self.image_encoder.encode(
                images).pooler_output  # img_encoded.shape = [batch_size, 2048, 1, 1]
            # img_encoded = img_encoded.permute(0, 3, 2, 1)
            img_encoded = img_encoded.reshape(batch_size, 1, -1)

        # Scale the output into [-1, 1]
        img_encoded = 2 * img_encoded - 1

        if img_encoded.ndim == 2:
            img_encoded = img_encoded.unsqueeze(1)
        return img_encoded

    def read_encode_images(self, num_samples_or_image):
        if isinstance(num_samples_or_image, str):
            sketch_path = DATA_DIR / num_samples_or_image
            img = Image.open(sketch_path).convert('RGB')
            img = self.transform_train(img)
            num_samples_or_image = [img]
        if isinstance(num_samples_or_image, list) and isinstance(num_samples_or_image[0], str):
            new_num_samples_or_image = []
            for sample in num_samples_or_image:
                sketch_path = DATA_DIR / sample
                sketch_path = str(sketch_path)
                if sketch_path.endswith(".svg"):
                    svg = SVG(sketch_path)
                    img = svg_to_img(svg)
                else:
                    img = Image.open(sketch_path).convert('RGB')
                # img = Image.open(sketch_path).convert('RGB')
                img = self.transform_train(img)
                new_num_samples_or_image.append(img)
            num_samples_or_image = new_num_samples_or_image
        if isinstance(num_samples_or_image, int):
            batch_size = num_samples_or_image
            dataset_builder = DatasetBuilder(self.hparams)
            ds = dataset_builder.val_dataloader()
            images = [ds[i][1] for i in range(batch_size)]
        elif isinstance(num_samples_or_image, list):
            images = num_samples_or_image
            batch_size = len(num_samples_or_image)

        images = torch.stack(images, dim=0)
        img_encoded = self.image_encode(images, batch_size)

        img_encoded.to(self.device)
        images.to(self.device)
        return img_encoded, images

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

    def configure_optimizers(self) -> Tuple[List, List]:
        lr = self.hparams.lr

        params_list = []
        trainable_parameters = list(self.net.parameters())
        params_list.append({'params': trainable_parameters, 'lr': lr})

        if self.optimizer_type == 'muon':
            self.automatic_optimization = False
            muon_params = []
            adamw_params = []

            for name, param in self.net.named_parameters():
                if (param.ndim == 2 and
                        'token_embedding' not in name and
                        'norm' not in name and
                        param.requires_grad):
                    muon_params.append(param)
                else:
                    adamw_params.append(param)

            optimizer_muon = instantiate_from_config(self.optimizer_cfg['optimizer_muon'],
                                                     params=muon_params,
                                                     lr=lr)
            optimizer_adaw = instantiate_from_config(self.optimizer_cfg['optimizer_adaw'],
                                                     params=adamw_params,
                                                     lr=lr)

            optimizers = [optimizer_muon, optimizer_adaw]

        else:  # adamw
            optimizer_adaw = instantiate_from_config(self.optimizer_cfg['optimizer_adaw'],
                                                     params=params_list, lr=lr)
            optimizers = [optimizer_adaw]

        # optimizer = instantiate_from_config(self.optimizer_cfg['optimizer'], params=params_list, lr=lr)
        if hasattr(self.optimizer_cfg, 'scheduler'):
            scheduler_func = instantiate_from_config(
                self.optimizer_cfg.scheduler,
                max_decay_steps=self.trainer.max_steps,
                lr_max=lr
            )

            schedulers = []
            for optimizer in optimizers:
                scheduler = {
                    "scheduler": lr_scheduler.LambdaLR(optimizer, lr_lambda=scheduler_func.schedule),
                    "interval": "step",
                    "frequency": 1
                }
                schedulers.append(scheduler)

        else:
            schedulers = []

        return optimizers, schedulers

    def get_sigmas(self, timesteps: Tensor, n_dim: int, dtype=torch.float32):
        sigmas = self.scheduler.sigmas.to(dtype=dtype, device=device)
        schedule_timesteps = self.scheduler.timesteps.to(device)
        timesteps = timesteps.to(device)

        step_indices = [(schedule_timesteps == t).nonzero()[0].item() for t in timesteps]

        sigma = sigmas[step_indices].flatten()
        while len(sigma.shape) < n_dim:
            sigma = sigma.unsqueeze(-1)
        return sigma

    def generate_null_img_encoded(self, image_shape, batch_size):
        seed = 2025
        if self.hparams.get('seed'):
            seed = self.hparams.get('seed')
        generator = torch.manual_seed(seed)

        dtype = torch.float32
        if self.device.type == "mps":
            # randn does not work reproducibly on mps
            image = randn_tensor(image_shape, generator=generator, dtype=dtype)
        else:
            image = randn_tensor(
                image_shape,
                generator=generator,
                device=self.device,
                dtype=dtype,
            )
        # convert to [0, 1]
        normalized_tensor = torch.sigmoid(image)
        null_img_encoded = normalized_tensor.to(self.device)

        return null_img_encoded

    @contextmanager
    def ema_scope(self, context=None):
        if self.ema_config is not None and self.ema_config.get('ema_inference', False):
            self.model_ema.store(self.net)
            self.model_ema.copy_to(self.net)
            if context is not None:
                print(f"{context}: Switched to EMA weights")
        try:
            yield None
        finally:
            if self.ema_config is not None and self.ema_config.get('ema_inference', False):
                self.model_ema.restore(self.net)
                if context is not None:
                    print(f"{context}: Restored training weights")

    def on_train_batch_end(self, *args, **kwargs):
        if self.ema_config is not None:
            self.model_ema(self.net)

    @torch.no_grad()
    def sample(
            self,
            num_samples_or_image,
            return_traj=False,
            return_cond=False,
            classifier_free_guidance=False,
            free_guidance_weight=1.0,
            num_inference_steps=50,
            sigmas: List[float] = None,
    ):
        img_encoded, images = self.read_encode_images(num_samples_or_image)
        images_shape = images.shape

        if self.hparams.get("use_zc"):
            # x_T = torch.randn([batch_size, 16, 512]).to(self.device)
            num_tokens = 16
            num_channels_latents = 512
        else:
            # x_T = torch.randn([batch_size, 16, 16]).to(self.device)
            num_tokens = 16
            num_channels_latents = 16

        latents = self.pipeline(img_encoded,
                                images_shape,
                                num_inference_steps=num_inference_steps,
                                num_tokens=num_tokens,
                                num_channels_latents=num_channels_latents,
                                guidance_scale=free_guidance_weight,
                                stage=1)
        return latents

    @torch.no_grad()
    def sampling_gaussians(
            self,
            num_samples_or_image,
            classifier_free_guidance=False,
            free_guidance_weight=1.0,
            return_cond=False,
            num_inference_steps=50,
    ):
        with self.ema_scope("Sample"):
            # with torch.amp.autocast(device_type='cuda'):
            gaus = self.sample(
                num_samples_or_image,
                classifier_free_guidance=classifier_free_guidance,
                free_guidance_weight=free_guidance_weight,
                num_inference_steps=num_inference_steps,
                return_cond=return_cond,
            )
            if isinstance(gaus, tuple):
                image = gaus[1]
                gaus = gaus[0]
            if self.hparams.get("global_normalization"):
                if not hasattr(self, "data_val"):
                    if self.hparams.spaghetti_tag == 'chairs_large':
                        from src.dataset_preparation.spaghetti_dataset import DatasetBuilder
                    else:
                        from src.dataset_preparation.spaghetti_dataset_v3 import DatasetBuilder
                    dataset_builder = DatasetBuilder(self.hparams)
                    ds = dataset_builder.build_dataset('val')
                if self.hparams.get("global_normalization") == "partial":
                    gaus = ds.unnormalize_global_static(gaus, slice(12, None))
                elif self.hparams.get("global_normalization") == "all":
                    gaus = ds.unnormalize_global_static(gaus, slice(None))

            gaus = project_eigenvectors(clip_eigenvalues(gaus))
            if return_cond:
                return gaus, image
            return gaus

    @torch.no_grad()
    def validation(self):
        test_sketch_path = DATA_DIR / self.hparams.dataset_kwargs.test_sketch_path
        test_images = get_test_images(test_sketch_path)
        extrinsics = self.sampling_gaussians(test_images,
                                             classifier_free_guidance=self.classifier_free_guidance,
                                             free_guidance_weight=self.free_guidance_weight,
                                             num_inference_steps=self.num_inference_steps
                                             )

        k = 256
        image_res = (k, k)

        image_transform = transforms.Compose([transforms.Resize(image_res), ])
        all_img = []
        for i, x in enumerate(extrinsics):
            sketch_path = test_images[i]
            sketch_path = str(sketch_path)
            if sketch_path.endswith(".svg"):
                svg = SVG(sketch_path)
                sketch_img = svg_to_img(svg)
            else:
                sketch_img = Image.open(sketch_path).convert('RGB')
            # sketch_img = Image.open(DATA_DIR / sketch_path).convert('RGB')
            sketch_img = image_transform(sketch_img)

            gaus_img = render_gaussians(x, resolution=image_res)
            img = merge_images([sketch_img, gaus_img])
            all_img.append(img)

        if all_img:
            try:
                self.logger.log_image('extrinsics', all_img)
            except Exception as e:
                logger.error(f"Wandb Logging failed: {e}")
        else:
            logger.error(f"No rendered images for 3D generation to log!!!")
