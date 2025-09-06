# -*- coding: UTF-8 -*-
"""
@Time : 16/06/2025 19:31
@Author : Xiaoguang Liang
@File : train.py
@Project : Enhancing_Sketch-to-3D_Controllability
"""
import os
import gc
import warnings

import hydra
import torch
import pytorch_lightning as pl
from omegaconf import OmegaConf
from sentry_sdk import capture_exception
from pytorch_lightning.strategies import DDPStrategy, DeepSpeedStrategy
from pytorch_lightning.utilities import rank_zero_info

from src.utils.sys_util import print_config
from configs.log_config import logger
from configs.global_setting import DATA_DIR
from src.utils.common import timer

torch._dynamo.config.cache_size_limit = 256

warnings.filterwarnings("ignore", category=FutureWarning)
torch.set_warn_always(False)

# train_file_name = 'train'
# train_file_name = 'train_debug'
# train_file_name = 'train_multirun'
# train_file_name = 'train_spaghetti'
# train_file_name = 'train_shape_vae'
# train_file_name = 'train_flow_diffusion'
# train_file_name = 'train_flow_diffusion_3ddit'
train_file_name = 'train_spaghetti_rectified_flow'

config_name = f'{train_file_name}.yaml'


@hydra.main(config_path="configs", config_name=config_name)
def main(config):
    if config.fast:
        torch.backends.cudnn.allow_tf32 = True
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.set_float32_matmul_precision('medium')
        torch.utils.data._utils.MP_STATUS_CHECK_INTERVAL = 0.05

    pl.seed_everything(config.seed)
    print_config(config, ("data", "model", "callbacks", "logger", "trainer", "paths", "hydra", "debug"))

    # Build model
    if train_file_name in ['train_flow_diffusion', 'train_shape_vae', 'train_flow_diffusion_3ddit',
                           'train_flow_ldm']:
        model = hydra.utils.instantiate(config.model.model)
    else:
        model = hydra.utils.instantiate(config.model)

    nodes = config.num_nodes
    ngpus = config.num_gpus
    base_lr = config.model.lr
    accumulate_grad_batches = config.update_every
    batch_size = config.batch_size

    if 'NNODES' in os.environ:
        nodes = int(os.environ['NNODES'])
        config.num_nodes = nodes

    if config.scale_lr:
        model.learning_rate = accumulate_grad_batches * nodes * ngpus * batch_size * base_lr
        info = f"Setting learning rate to {model.learning_rate:.2e} = {accumulate_grad_batches} (accumulate)"
        info += f" * {nodes} (nodes) * {ngpus} (num_gpus) * {batch_size} (batchsize) * {base_lr:.2e} (base_lr)"
        rank_zero_info(info)
    else:
        model.learning_rate = base_lr
        rank_zero_info("++++ NOT USING LR SCALING ++++")
        rank_zero_info(f"Setting learning rate to {model.learning_rate:.2e}")

    # Build trainer
    if config.num_nodes > 1 or config.num_gpus > 1:
        if config.deepspeed:
            ddp_strategy = DeepSpeedStrategy(stage=1)
        elif config.deepspeed2:
            ddp_strategy = 'deepspeed_stage_2'
        else:
            ddp_strategy = DDPStrategy(find_unused_parameters=False, bucket_cap_mb=1500)
    else:
        ddp_strategy = 'auto'

    # logger.info(f'*' * 100)
    # if config.use_amp:
    #     amp_type = config.amp_type
    #     assert amp_type in ['bf16', '16', '32'], f"Invalid amp_type: {amp_type}"
    #     rank_zero_info(f'Using {amp_type} precision')
    # else:
    #     amp_type = 32
    #     rank_zero_info(f'Using 32 bit precision')
    # rank_zero_info(f'*' * 100)

    # Save model config
    model_config_path = os.path.join(config.paths.save_dir, 'hparams.yaml')
    logger.info(f'Save model config to {model_config_path}')
    with open(model_config_path, 'w') as f:
        if train_file_name in ['train_flow_diffusion', 'train_shape_vae']:
            OmegaConf.save(config=config.model.model, f=f, resolve=True)
        else:
            OmegaConf.save(config=config.model, f=f, resolve=True)

    callbacks = []
    if config.get("callbacks"):
        for cb_name, cb_conf in config.callbacks.items():
            if config.get("debug") and cb_name == "model_checkpoint":
                continue
            callbacks.append(hydra.utils.instantiate(cb_conf))

    logger_config = []
    if config.get("logger"):
        for lg_name, lg_conf in config.logger.items():
            log_model = hydra.utils.instantiate(lg_conf)
            if lg_name == "wandb":
                log_model.watch(model, log="all", log_freq=5000, log_graph=False)
            logger_config.append(log_model)

    trainer = hydra.utils.instantiate(
        config.trainer,
        strategy=ddp_strategy,
        callbacks=callbacks,
        logger=logger_config if len(logger_config) != 0 else False,
        _convert_="partial",
        precision=config.precision,
    )

    logger.info(f'Start training {config.model_name}')
    # Build data modules
    if 'flow_diffusion' in config.model_name:
        config.dataset.image_path = str(DATA_DIR / config.dataset.image_path)
        config.dataset.train_data_list = str(DATA_DIR / config.dataset.train_data_list)
        config.dataset.val_data_list = str(DATA_DIR / config.dataset.val_data_list)
        data: pl.LightningDataModule = hydra.utils.instantiate(config.dataset)
        trainer.fit(model, datamodule=data)
    elif config.model_name in ['shape_vae']:
        config.dataset.train_data_list = str(DATA_DIR / config.dataset.train_data_list)
        config.dataset.val_data_list = str(DATA_DIR / config.dataset.val_data_list)
        data: pl.LightningDataModule = hydra.utils.instantiate(config.dataset)
        trainer.fit(model, datamodule=data)
    else:
        if config.model.dataset_kwargs.spaghetti_tag == 'chairs_large':
            from src.dataset_preparation.spaghetti_dataset import DatasetBuilder
        else:
            # from src.dataset_preparation.spaghetti_dataset_v3 import DatasetBuilder
            from src.dataset_preparation.spaghetti_dataset_v4 import DatasetBuilder
        dataset_builder = DatasetBuilder(config.model)
        train_data = dataset_builder.train_dataloader()
        val_data = dataset_builder.val_dataloader()
        trainer.fit(model, train_dataloaders=train_data, val_dataloaders=val_data,
                    ckpt_path=config.resume_checkpoint_path)

    if "stage2" in config.model_name:
        from src.metrics.evaluate_model import evaluate_model

        logger.info("Start evaluating model")
        test_3d_path = DATA_DIR / config.model.dataset_kwargs.test_3d_path
        test_sketch_path = DATA_DIR / config.model.dataset_kwargs.test_sketch_path
        with timer("Evaluation time"):
            results = evaluate_model(model, test_3d_path, test_sketch_path, num_samples=None,
                                     batch_size=config.model.batch_size, pattern="*.jpg")
            model.logger.log_metrics(results)

    # Release the GPU memory to avoid OOM
    torch.cuda.empty_cache()
    del model
    del trainer
    gc.collect()


if __name__ == "__main__":
    try:
        with timer("Total time"):
            main()
    except Exception as exc:
        capture_exception(exc)
        logger.error(exc)
        raise exc
