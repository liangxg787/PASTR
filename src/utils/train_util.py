# -*- coding: UTF-8 -*-
"""
@Time : 16/06/2025 18:25
@Author : Xiaoguang Liang
@File : train_util.py
@Project : Enhancing_Sketch-to-3D_Controllability
"""
from typing import Literal
import math

import torch
from omegaconf import OmegaConf

from configs.global_setting import BASE_DIR


class PolyDecayScheduler(torch.optim.lr_scheduler.LambdaLR):
    def __init__(self, optimizer, init_lr, power=0.99, lr_end=1e-7, last_epoch=-1):
        def lr_lambda(step):
            lr = max(power ** step, lr_end / init_lr)
            return lr

        super().__init__(optimizer, lr_lambda, last_epoch=last_epoch)


def get_dropout_mask(shape, dropout: float, device):
    if dropout == 1:
        return torch.zeros(shape, device=device, dtype=torch.bool)
    elif dropout == 0:
        return torch.ones_like(shape, device=device, dtype=torch.bool)
    else:
        return torch.zeros(shape, device=device).float().uniform_(0, 1) > dropout


def load_model(time_folder, model, model_name: str, device, do_eval=True):
    # hparams_file = str(BASE_DIR / f"runs/{model_name}/{time_folder}/hparams.yaml")
    # c = OmegaConf.load(hparams_file)
    model = model.load_from_checkpoint(BASE_DIR / f"runs/{model_name}/{time_folder}/checkpoints/last.ckpt",
                                       map_location=device)

    if do_eval:
        model.eval()
        for p in model.parameters():
            p.requires_grad_(False)
    return model


def compute_density_for_timestep_sampling(
        weighting_scheme: str, batch_size: int, logit_mean: float = None, logit_std: float = None,
        mode_scale: float = None
):
    """
    Compute the density for sampling the timesteps when doing SD3 training.

    Courtesy: This was contributed by Rafie Walker in https://github.com/huggingface/diffusers/pull/8528.

    SD3 paper reference: https://arxiv.org/abs/2403.03206v1.
    """
    if weighting_scheme == "logit_normal":
        # See 3.1 in the SD3 paper ($rf/lognorm(0.00,1.00)$).
        u = torch.normal(mean=logit_mean, std=logit_std, size=(batch_size,), device="cpu")
        u = torch.nn.functional.sigmoid(u)
    elif weighting_scheme == "mode":
        u = torch.rand(size=(batch_size,), device="cpu")
        u = 1 - u - mode_scale * (torch.cos(math.pi * u / 2) ** 2 - 1 + u)
    else:
        u = torch.rand(size=(batch_size,), device="cpu")
    return u


def compute_loss_weighting_for_sd3(weighting_scheme: str, sigmas=None):
    """
    Computes loss weighting scheme for SD3 training.

    Courtesy: This was contributed by Rafie Walker in https://github.com/huggingface/diffusers/pull/8528.

    SD3 paper reference: https://arxiv.org/abs/2403.03206v1.
    """
    if weighting_scheme == "sigma_sqrt":
        weighting = (sigmas ** -2.0).float()
    elif weighting_scheme == "cosmap":
        bot = 1 - 2 * sigmas + 2 * sigmas ** 2
        weighting = 2 / (math.pi * bot)
    else:
        weighting = torch.ones_like(sigmas)
    return weighting
