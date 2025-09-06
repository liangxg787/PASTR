# -*- coding: UTF-8 -*-
"""
@Time : 16/06/2025 18:03
@Author : Xiaoguang Liang
@File : stage1.py
@Project : Enhancing_Sketch-to-3D_Controllability
"""
import random

import torch
import torch.nn.functional as F
from tqdm import trange
from PIL import Image
from diffusers.utils.torch_utils import randn_tensor
from torchvision import transforms

from src.models.base_model import BaseModel
from src.utils.image_util import merge_images, draw_text
from src.utils.visual_util import render_gaussians
from src.utils.spaghetti_util import clip_eigenvalues, project_eigenvectors
from src.models import init_model
from src.utils.image_util import build_transforms
from configs.global_setting import DATA_DIR
from configs.log_config import logger
from src.dataset_preparation.spaghetti_dataset import DatasetBuilder
from src.utils.data_util import get_test_images


class Stage1Model(BaseModel):
    def __init__(self, network, variance_schedule, **kwargs):
        super().__init__(network, variance_schedule, **kwargs)

        self.transform_train, self.transform_test = build_transforms(self.hparams.dataset_kwargs)
        self.image_encoder = init_model(self.hparams.encoder_model)

    def forward(self, x, img):
        """
        Input:
            x: [B,G,16]
            text: list of length [B]
        """
        B, G = x.shape[:2]
        img_encoded = self.image_encode(img, B)
        return self.get_loss(x, img_encoded)

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
                img = Image.open(sketch_path).convert('RGB')
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

    def step(self, batch, stage: str):
        x, text = batch
        loss = self(x, text)

        self.log(f"{stage}/loss", loss, on_step=stage == "train", prog_bar=True, on_epoch=True, logger=True)
        return loss

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
        image = normalized_tensor.to(self.device)

        null_img_encoded = self.image_encode(image, batch_size)
        return null_img_encoded

    @torch.no_grad()
    def sample(
            self,
            num_samples_or_image,
            return_traj=False,
            return_cond=False,
            classifier_free_guidance=True,
            free_guidance_weight=2.0,
    ):
        img_encoded, images = self.read_encode_images(num_samples_or_image)
        batch_size = images.shape[0]

        if self.hparams.get("use_zc"):
            x_T = torch.randn([batch_size, 16, 512]).to(self.device)
        else:
            x_T = torch.randn([batch_size, 16, 16]).to(self.device)
        G = x_T.shape[1]

        if classifier_free_guidance:
            image_shape = images.shape
            null_img_encoded = self.generate_null_img_encoded(image_shape, batch_size)

        traj = {self.var_sched.num_steps: x_T}
        logger.info(f"Start Stage1 sampling ...")
        for t in trange(self.var_sched.num_steps, 0, -1):
            z = torch.randn_like(x_T) if t > 1 else torch.zeros_like(x_T)
            alpha = self.var_sched.alphas[t]
            alpha_bar = self.var_sched.alpha_bars[t]
            sigma = self.var_sched.get_sigmas(t, flexibility=0)

            c0 = 1.0 / torch.sqrt(alpha)
            c1 = (1 - alpha) / torch.sqrt(1 - alpha_bar)

            x_t = traj[t]

            beta = self.var_sched.betas[[t] * batch_size]
            e_theta = self.net(x_t, beta=beta, context=img_encoded)

            if classifier_free_guidance:
                null_e_theta = self.net(x_t, beta=beta, context=null_img_encoded)
                w = free_guidance_weight
                e_theta = (1 + w) * e_theta - w * null_e_theta

            x_next = c0 * (x_t - c1 * e_theta) + sigma * z
            traj[t - 1] = x_next.detach()

            traj[t] = traj[t].cpu()

            if not return_traj:
                del traj[t]

        if return_traj:
            if return_cond:
                return traj, img_encoded
            return traj
        else:
            if return_cond:
                return traj[0], img_encoded
            return traj[0]

    def sampling_gaussians(
            self,
            num_samples_or_image,
            classifier_free_guidance=False,
            free_guidance_weight=2.0,
            return_cond=False,
    ):
        gaus = self.sample(
            num_samples_or_image,
            classifier_free_guidance=classifier_free_guidance,
            free_guidance_weight=free_guidance_weight,
            return_cond=return_cond,
        )
        if isinstance(gaus, tuple):
            image = gaus[1]
            gaus = gaus[0]
        if self.hparams.get("global_normalization"):
            if not hasattr(self, "data_val"):
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
        extrinsics = self.sampling_gaussians(test_images)

        k = 256
        image_res = (k, k)

        image_transform = transforms.Compose([transforms.Resize(image_res), ])
        all_img = []
        for i, x in enumerate(extrinsics):
            sketch_path = test_images[i]
            sketch_img = Image.open(DATA_DIR / sketch_path).convert('RGB')
            sketch_img = image_transform(sketch_img)

            gaus_img = render_gaussians(x, resolution=image_res)
            img = merge_images([sketch_img, gaus_img])
            all_img.append(img)

        self.logger.log_image('extrinsics', all_img)
