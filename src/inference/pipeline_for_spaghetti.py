# -*- coding: UTF-8 -*-
"""
@Time : 16/08/2025 00:04
@Author : Xiaoguang Liang
@File : pipeline_for_spaghetti.py
@Project : Enhancing_Sketch-to-3D_Controllability
"""
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import torch
from diffusers.schedulers import FlowMatchEulerDiscreteScheduler
from diffusers.pipelines.pipeline_utils import DiffusionPipeline
from diffusers.utils.torch_utils import randn_tensor
import PIL
import PIL.Image

from src.inference.pipelines import retrieve_timesteps
from configs.global_setting import device
from src.utils.common import np2th


class RectifiedFlowPipeline(DiffusionPipeline):
    """
    Pipeline for image to 3D part-level object generation.
    """

    def __init__(
            self,
            model,
            scheduler: FlowMatchEulerDiscreteScheduler,
            hparams,
    ):
        super().__init__()

        self.register_modules(
            model=model,
            scheduler=scheduler,
            hparams=hparams,
        )

    @property
    def guidance_scale(self):
        return self._guidance_scale

    @property
    def do_classifier_free_guidance(self):
        return self._guidance_scale > 1

    @property
    def num_timesteps(self):
        return self._num_timesteps

    @property
    def attention_kwargs(self):
        return self._attention_kwargs

    @property
    def interrupt(self):
        return self._interrupt

    @property
    def decode_progressive(self):
        return self._decode_progressive

    def prepare_latents(
            self,
            batch_size,
            num_tokens,
            num_channels_latents,
            dtype,
            device,
            generator,
            latents: Optional[torch.Tensor] = None,
    ):
        shape = (batch_size, num_tokens, num_channels_latents)

        if isinstance(generator, list) and len(generator) != batch_size:
            raise ValueError(
                f"You have passed a list of generators of length {len(generator)}, but requested an effective batch"
                f" size of {batch_size}. Make sure the batch size matches the length of the generators."
            )

        noise = randn_tensor(shape, generator=generator, device=device, dtype=dtype)
        return noise

    def generate_null_img_encoded(self, image_shape, batch_size):
        seed = 2025
        if self.hparams.get('seed'):
            seed = self.hparams.get('seed')
        generator = torch.manual_seed(seed)

        dtype = torch.float32
        if device.type == "mps":
            # randn does not work reproducibly on mps
            image = randn_tensor(image_shape, generator=generator, dtype=dtype)
        else:
            image = randn_tensor(
                image_shape,
                generator=generator,
                device=device,
                dtype=dtype,
            )
        # convert to [0, 1]
        normalized_tensor = torch.sigmoid(image)
        image = normalized_tensor.to(device)

        # null_img_encoded = self.image_encode(image, batch_size)
        img_encoded = image.reshape(batch_size, 1, -1)

        # Scale the output into [-1, 1]
        null_img_encoded = 2 * img_encoded - 1

        if img_encoded.ndim == 2:
            null_img_encoded = img_encoded.unsqueeze(1)
        return null_img_encoded

    def generate_null_cond(self, B, G, image_shape):
        null_img_encoded = self.generate_null_img_encoded(image_shape, B)

        gaus = torch.zeros(B, G, 16, dtype=torch.float, device=device)
        return self.cond_from_gaus_img(gaus, null_img_encoded)

    def cond_from_gaus_img(self, gaus, img_encoded):
        gaus = np2th(gaus).to(device)
        G = gaus.shape[1]
        img_encoded = np2th(img_encoded).to(device)
        assert gaus.ndim == 3
        if img_encoded.ndim == 2:
            img_encoded = img_encoded.unsqueeze(1)
        img_encoded = img_encoded.expand(-1, G, -1)
        return torch.cat([gaus, img_encoded], -1)

    @torch.no_grad()
    def __call__(
            self,
            img_encoded,
            images_shape,
            num_inference_steps: int = 50,
            num_tokens: int = 16,
            num_channels_latents: int = 16,
            timesteps: List[int] = None,
            guidance_scale: float = 1,
            generator: Optional[Union[torch.Generator, List[torch.Generator]]] = None,
            latents: Optional[torch.FloatTensor] = None,
            attention_kwargs: Optional[Dict[str, Any]] = None,
            callback_on_step_end: Optional[Callable[[int, int, Dict], None]] = None,
            callback_on_step_end_tensor_inputs: List[str] = ["latents"],
            stage=1,
            cond=None,
    ):
        # 1. Define call parameters
        self._guidance_scale = guidance_scale
        self._attention_kwargs = attention_kwargs
        self._interrupt = False

        # 2. Define call parameters
        batch_size = img_encoded.shape[0]

        image_shape = img_encoded.shape

        if self.do_classifier_free_guidance:
            if stage == 1:

                null_cond = self.generate_null_img_encoded(image_shape, batch_size)
            elif stage == 2:
                G = cond.shape[1]
                null_cond = self.generate_null_cond(batch_size, G, image_shape)

        # 4. Prepare timesteps
        timesteps, num_inference_steps = retrieve_timesteps(
            self.scheduler, num_inference_steps, device, timesteps
        )
        num_warmup_steps = max(
            len(timesteps) - num_inference_steps * self.scheduler.order, 0
        )
        self._num_timesteps = len(timesteps)

        # 5. Prepare latent variables
        latents = self.prepare_latents(
            batch_size,
            num_tokens,
            num_channels_latents,
            img_encoded.dtype,
            device,
            generator,
            latents,
        )

        # 6. Denoising loop
        self.set_progress_bar_config(
            desc="Denoising",
            ncols=125,
            disable=self._progress_bar_config['disable'] if hasattr(self, '_progress_bar_config') else False,
        )

        if stage == 1:
            cond = img_encoded

        with self.progress_bar(total=len(timesteps)) as progress_bar:
            for i, t in enumerate(timesteps):
                if self.interrupt:
                    continue

                timestep = t.expand(latents.shape[0])

                noise_pred = self.model(
                    latents,
                    beta=timestep,
                    context=cond,
                )

                if self.do_classifier_free_guidance:
                    null_e_theta = self.model(latents, beta=timestep, context=null_cond)
                    w = self._guidance_scale
                    noise_pred = (1 + w) * noise_pred - w * null_e_theta

                # compute the previous noisy sample x_t -> x_t-1
                latents_dtype = latents.dtype
                latents = self.scheduler.step(
                    noise_pred, t, latents, return_dict=False
                )[0]

                if latents.dtype != latents_dtype:
                    if torch.backends.mps.is_available():
                        # some platforms (eg. apple mps) misbehave due to a pytorch bug: https://github.com/pytorch/pytorch/pull/99272
                        latents = latents.to(latents_dtype)

                if callback_on_step_end is not None:
                    callback_kwargs = {}
                    for k in callback_on_step_end_tensor_inputs:
                        callback_kwargs[k] = locals()[k]
                    callback_outputs = callback_on_step_end(self, i, t, callback_kwargs)

                # call the callback, if provided
                if i == len(timesteps) - 1 or (
                        (i + 1) > num_warmup_steps and (i + 1) % self.scheduler.order == 0
                ):
                    progress_bar.update()
        return latents
