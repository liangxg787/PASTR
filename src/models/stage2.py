# -*- coding: UTF-8 -*-
"""
@Time : 16/06/2025 18:03
@Author : Xiaoguang Liang
@File : stage2.py
@Project : Enhancing_Sketch-to-3D_Controllability
"""
import random

from tqdm import trange
import torch
import torch.nn.functional as F
from PIL import Image
from torchvision import transforms

from src.models.stage1 import Stage1Model
from src.utils.image_util import merge_images, draw_text
from src.utils.visual_util import render_gaussians, render_mesh
from src.utils.common import np2th
from src.utils.spaghetti_util import (generate_zc_from_sj_gaus,
                                      get_mesh_from_spaghetti, load_marching_cube_meshing,
                                      load_spaghetti)
from src.utils.train_util import get_dropout_mask, load_model
from configs.global_setting import DATA_DIR, device
from configs.log_config import logger
from src.dataset_preparation.spaghetti_dataset import DatasetBuilder
from src.utils.data_util import get_test_images


class Stage2Model(Stage1Model):
    def __init__(self, network, variance_schedule, **kwargs):
        super().__init__(network, variance_schedule, **kwargs)

        if not self.hparams.no_run_validation:
            # Initialize mesher
            self.mc_mesher = load_marching_cube_meshing(device=device)
            # Load stage1 model
            time_folder = self.hparams.stage1_model_path
            self.stage1_model = load_model(time_folder, Stage1Model, "stage1", device)

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
        B, G, D = x0.shape
        if not noisy_in:
            if t is None:
                t = self.var_sched.uniform_sample_t(B)
            x_noisy, beta, e_rand = self.add_noise(x0, t)
        else:
            x_noisy = x0
            beta = beta_in
            e_rand = e_rand_in
        e_theta = self.net(x_noisy, beta, cond)
        loss = F.mse_loss(e_theta.flatten(), e_rand.flatten(), reduction="mean")
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
    ):

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

        G = cond.shape[1]
        if classifier_free_guidance:
            null_cond = self.generate_null_cond(batch_size, G, image_shape)

        x_T = torch.randn([batch_size, 16, 512]).to(self.device)
        traj = {self.var_sched.num_steps: x_T}
        logger.info(f"Start Stage2 sampling ...")
        for t in trange(self.var_sched.num_steps, 0, -1):
            z = torch.randn_like(x_T) if t > 1 else torch.zeros_like(x_T)
            alpha = self.var_sched.alphas[t]
            alpha_bar = self.var_sched.alpha_bars[t]
            sigma = self.var_sched.get_sigmas(t, flexibility=0)

            c0 = 1.0 / torch.sqrt(alpha)
            c1 = (1 - alpha) / torch.sqrt(1 - alpha_bar)

            x_t = traj[t]

            beta = self.var_sched.betas[[t] * batch_size]
            e_theta = self.net(x_t, beta=beta, context=cond)

            if classifier_free_guidance:
                null_e_theta = self.net(x_t, beta=beta, context=null_cond)
                w = free_guidance_weight
                e_theta = (1 + w) * e_theta - w * null_e_theta

            x_next = c0 * (x_t - c1 * e_theta) + sigma * z
            traj[t - 1] = x_next.detach()

            traj[t] = traj[t].cpu()

            if not return_traj:
                del traj[t]

        if return_traj:
            if return_cond:
                return traj, cond
            return traj
        else:
            if return_cond:
                return traj[0], cond
            return traj[0]

    @torch.no_grad()
    def validation(self):
        # Load spaghetti
        spaghetti = load_spaghetti(device=device, tag=self.hparams.spaghetti_tag)
        test_sketch_path = DATA_DIR / self.hparams.dataset_kwargs.test_sketch_path
        test_images = get_test_images(test_sketch_path)

        # Stage1 sampling
        extrinsics = self.stage1_model.sampling_gaussians(test_images)
        # stage2 sampling
        intrinsics = self.sample(test_images, extrinsics)

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
            self.logger.log_image('Generated results', all_img)
