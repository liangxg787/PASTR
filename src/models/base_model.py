# -*- coding: UTF-8 -*-
"""
@Time : 16/06/2025 18:04
@Author : Xiaoguang Liang
@File : base_model.py
@Project : Enhancing_Sketch-to-3D_Controllability
"""
import torch
import torch.nn.functional as F
import pytorch_lightning as pl
from diffusers.optimization import get_cosine_schedule_with_warmup

from configs.log_config import logger
from src.utils.train_util import PolyDecayScheduler


class BaseModel(pl.LightningModule):
    def __init__(
            self,
            network,
            variance_schedule,
            **kwargs,
    ):
        super().__init__()
        self.save_hyperparameters()
        self.net = network
        self.var_sched = variance_schedule

    def forward(self, x):
        return self.get_loss(x)

    def step(self, x, stage: str):
        loss = self(x)
        self.log(
            f"{stage}/loss",
            loss,
            on_step=stage == "train",
            prog_bar=True,
        )
        return loss

    def training_step(self, batch, batch_idx):
        x = batch
        return self.step(x, "train")

    def add_noise(self, x, t):
        """
        Input:
            x: [B,D] or [B,G,D]
            t: list of size B
        Output:
            x_noisy: [B,D]
            beta: [B]
            e_rand: [B,D]
        """
        alpha_bar = self.var_sched.alpha_bars[t]
        beta = self.var_sched.betas[t]

        c0 = torch.sqrt(alpha_bar).view(-1, 1)  # [B,1]
        c1 = torch.sqrt(1 - alpha_bar).view(-1, 1)

        e_rand = torch.randn_like(x)
        if e_rand.dim() == 3:
            c0 = c0.unsqueeze(1)
            c1 = c1.unsqueeze(1)

        x_noisy = c0 * x + c1 * e_rand

        return x_noisy, beta, e_rand

    def get_loss(
            self,
            x0,
            t=None,
            noisy_in=False,
            beta_in=None,
            e_rand_in=None,
    ):
        if x0.dim() == 2:
            B, D = x0.shape
        else:
            B, G, D = x0.shape
        if not noisy_in:
            if t is None:
                t = self.var_sched.uniform_sample_t(B)
            x_noisy, beta, e_rand = self.add_noise(x0, t)
        else:
            x_noisy = x0
            beta = beta_in
            e_rand = e_rand_in

        e_theta = self.net(x_noisy, beta=beta)
        loss = F.mse_loss(e_theta.flatten(), e_rand.flatten(), reduction="mean")
        return loss

    @torch.no_grad()
    def sample(
            self,
            batch_size=0,
            return_traj=False,
    ):
        raise NotImplementedError

    def validation_step(self, batch, batch_idx):
        return

    def on_validation_epoch_end(self):
        if self.hparams.no_run_validation:
            return
        # if not self.trainer.sanity_checking:
        logger.info("Start evaluating model at N epochs end ...")
        self.validation()

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.hparams.lr)
        # scheduler = PolyDecayScheduler(optimizer, self.hparams.lr, power=0.999)
        # Get all the steps when training
        total_steps = self.trainer.estimated_stepping_batches
        scheduler = get_cosine_schedule_with_warmup(
            optimizer=optimizer,
            num_warmup_steps=self.hparams.lr_warmup_steps,
            num_training_steps=total_steps
        )
        lr_scheduler = {
            'scheduler': scheduler,
            'name': 'learning_rate'
        }
        return [optimizer], [lr_scheduler]
