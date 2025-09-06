# -*- coding: UTF-8 -*-
"""
@Time : 16/06/2025 18:12
@Author : Xiaoguang Liang
@File : make_input_dataset.py
@Project : Enhancing_Sketch-to-3D_Controllability
"""
import os
import json
import random
from fnmatch import fnmatch
from collections import defaultdict

import h5py
import torch
import numpy as np
from dotmap import DotMap
from PIL import Image
from omegaconf import OmegaConf, listconfig

from configs.global_setting import DATA_DIR
from src.utils.common import th2np
from src.utils.image_util import svg_to_img, SVG, build_transforms
from configs.log_config import logger
from src.utils.data_util import get_sketch_data, down_sample

# img_num = 6
img_num = 24


class ShapNetDataset(torch.utils.data.Dataset):
    def __init__(self, data_path, repeat=None, **kwargs):
        super().__init__()
        self.data_path = str(DATA_DIR / data_path)
        self.repeat = repeat
        self.mesh_names = []
        self.__dict__.update(kwargs)
        self.hparams = DotMap(self.__dict__)

        """
        Global Data statistics.
        """
        if self.hparams.get("global_normalization"):
            # mean_std_data_path = self.data_path.replace(".hdf5", "_mean_std.hdf5")
            mean_std_data_path = self.data_path
            with h5py.File(mean_std_data_path) as f:
                self.global_mean = f["mean"][:].astype(np.float32)
                logger.info(f"global_mean shape: {self.global_mean.shape}")
                self.global_std = f["std"][:].astype(np.float32)
                logger.info(f"global_std shape: {self.global_std.shape}")

        self.data = dict()

        with h5py.File(self.data_path) as f:
            if "mesh_names" in f.keys():
                self.mesh_names = list(f["mesh_names"][:].astype(str))
                # Make sure mesh_names is a list
                if isinstance(self.mesh_names, (list, np.ndarray)):
                    # If the element is bytes, convert it to string
                    if isinstance(self.mesh_names[0], (np.str_, np.bytes_)):
                        self.mesh_names = [str(name) for name in self.mesh_names]
                # self.mesh_names = list(
                # filter(lambda x: x not in ['179b88264e7f96468b442b160bcfb7fd1'], self.mesh_names))
            else:
                if 'salad' in self.data_path:
                    salad_ids_path = DATA_DIR / "salad_data/ids.json"
                    with open(salad_ids_path, "r") as fp:
                        self.mesh_names = json.load(fp)  # The amount of meshes: 2816

            for k in self.hparams.data_keys:
                self.data[k] = f[k][:].astype(np.float32)
                logger.info(f"{k} shape: {self.data[k].shape}")

                """
                global_normalization arg is for gaussians only.
                """
                if k == "g_js_affine":
                    if self.hparams.get("global_normalization") == "partial":
                        assert k == "g_js_affine"
                        if self.hparams.get("verbose"):
                            print("[*] Normalize data only for pi and eigenvalues.")
                        # 3: mu, 9: eigvec, 1: pi, 3: eigval
                        self.data[k] = self.normalize_global_static(
                            self.data[k], slice(12, None)
                        )
                    elif self.hparams.get("global_normalization") == "all":
                        assert k == "g_js_affine"
                        if self.hparams.get("verbose"):
                            print("[*] Normalize data for all elements.")
                        self.data[k] = self.normalize_global_static(
                            self.data[k], slice(None)
                        )

    def __getitem__(self, idx):
        if self.repeat is not None and self.repeat > 1:
            idx = int(idx / self.repeat)

        items = []
        for k in self.hparams.data_keys:
            data = torch.from_numpy(self.data[k][idx])
            items.append(data)

        if self.hparams.get("concat_data"):
            return torch.cat(items, -1)  # [16,528]
        if len(items) == 1:
            return items[0]
        return items

    def __len__(self):
        k = self.hparams.data_keys[0]
        if self.repeat is not None and self.repeat > 1:
            return len(self.data[k]) * self.repeat
        return len(self.data[k])

    def get_other_latents(self, key):
        with h5py.File(self.data_path) as f:
            return f[key][:].astype(np.float32)

    def normalize_global_static(self, data: np.ndarray, normalize_indices=slice(None)):
        """
        Input:
            np.ndarray or torch.Tensor. [16,16] or [B,16,16]
            slice(None) -> full
            slice(12, None) -> partial
        Output:
            [16,16] or [B,16,16]
        """
        assert normalize_indices == slice(None) or normalize_indices == slice(
            12, None
        ), print(f"{normalize_indices} is wrong.")
        data = th2np(data).copy()
        data[..., normalize_indices] = (
                                               data[..., normalize_indices] - self.global_mean[
                                           normalize_indices]
                                       ) / self.global_std[normalize_indices]
        return data

    def unnormalize_global_static(
            self, data: np.ndarray, unnormalize_indices=slice(None)
    ):
        """
        Input:
            np.ndarray or torch.Tensor. [16,16] or [B,16,16]
            slice(None) -> full
            slice(12, None) -> partial
        Output:
            [16,16] or [B,16,16]
        """
        assert unnormalize_indices == slice(None) or unnormalize_indices == slice(
            12, None
        ), print(f"{unnormalize_indices} is wrong.")
        data = th2np(data).copy()
        data[..., unnormalize_indices] = (
                                             data[..., unnormalize_indices]
                                         ) * self.global_std[unnormalize_indices] + self.global_mean[
                                             unnormalize_indices]
        return data

    def get_mesh_name(self, idx):
        # print('>' * 9, f"self.mesh_names: {self.mesh_names}")
        # print('>' * 9, f"self.mesh_names type: {type(self.mesh_names)}")
        if not self.mesh_names:
            raise ValueError("mesh_names is empty.")
        if isinstance(self.mesh_names, dict):
            idx = str(idx)
            mesh_name, spa_idx = self.mesh_names.get(idx, ['', ''])
            return mesh_name, spa_idx
        if isinstance(self.mesh_names, list):
            mesh_name = self.mesh_names[idx]
            return mesh_name, idx


class CLIPassoSketchDataset(ShapNetDataset):
    def __init__(self, data_path, repeat=None, **kwargs):
        super().__init__(data_path, repeat, **kwargs)

        self.sketch_data = get_sketch_data(DATA_DIR / self.hparams.sketch_path)
        self.sketch_keys = list(self.sketch_data.keys())
        self.transform_train, self.transform_test = build_transforms(self.hparams)

    def __getitem__(self, idx):
        if self.repeat is not None and self.repeat > 1:
            idx = idx // self.repeat

        mesh_name, spa_idx = self.get_mesh_name(idx)

        if mesh_name:
            sketch_paths = self.sketch_data[mesh_name]
            if self.repeat and self.repeat > 1:
                sketch_idx = int(idx % 6)
            else:
                # Get a random number from 0 to 6
                sketch_idx = np.random.randint(0, 6)

            sketch_path = sketch_paths[sketch_idx]
            svg = SVG(sketch_path)
            img = svg_to_img(svg)
            img = self.transform_train(img)

            latents = []
            for k in self.hparams.data_keys:
                data = torch.from_numpy(self.data[k][spa_idx])
                latents.append(data)

            item = latents + [img]
            if self.hparams.get("concat_data"):
                latents = torch.cat(latents, -1)
                return latents, img

            return item

    def __len__(self):
        # if self.repeat is not None and self.repeat > 1:
        #     return len(self.sketch_data) * self.repeat
        # return len(self.sketch_data)
        if self.repeat is not None and self.repeat > 1:
            return len(self.data['g_js_affine']) * self.repeat
        return len(self.data['g_js_affine'])


class InformativeSketchDataset(CLIPassoSketchDataset):
    def __init__(self, data_path, repeat=None, **kwargs):
        super().__init__(data_path, repeat, **kwargs)

        self.sketch_data = get_sketch_data(DATA_DIR / self.hparams.sketch_path, pattern="*.png")
        self.sketch_keys = list(self.sketch_data.keys())

    def __getitem__(self, idx):
        new_idx = idx
        if self.repeat is not None and self.repeat > 1:
            new_idx = idx // self.repeat

        mesh_name, spa_idx = self.get_mesh_name(new_idx)

        if mesh_name:
            sketch_paths = self.sketch_data[mesh_name]
            if not sketch_paths:
                # logger.error(f"No sketch found for {mesh_name}")
                mesh_name = random.choice(self.sketch_keys)
                sketch_paths = self.sketch_data[mesh_name]
        else:
            keys = list(self.sketch_data.keys())
            mesh_name = random.choice(keys)
            sketch_paths = self.sketch_data[mesh_name]

        if self.repeat and self.repeat > 1:
            sketch_idx = int(idx % img_num)
        else:
            # Get a random number from 0 to 6
            sketch_idx = np.random.randint(0, img_num)

        sketch_path = sketch_paths[sketch_idx]
        img = Image.open(sketch_path).convert('RGB')
        img = self.transform_train(img)  # img.shape = [3, 256, 256]

        latents = []
        for k in self.hparams.data_keys:
            data = torch.from_numpy(self.data[k][spa_idx])
            latents.append(data)

        item = latents + [img]
        if self.hparams.get("concat_data"):
            latents = torch.cat(latents, -1)
            return latents, img

        return item


class AllDataset(ShapNetDataset):

    def __init__(self, data_path, repeat=None, **kwargs):
        super().__init__(data_path, repeat, **kwargs)

        self.sketch_data = self.merge_sketch_data(self.hparams.sketch_path, self.hparams.sketch_tag)
        self.sketch_keys = list(self.sketch_data.keys())
        self.transform_train, self.transform_test = build_transforms(self.hparams)

    def merge_sketch_data(self, sketch_path, sketch_tag):
        data_dict_list = []
        fields = ['data_sketch1', 'data_sketch2', 'data_sketch3', 'data_sketch4', 'data_sketch5']
        if isinstance(sketch_path, listconfig.ListConfig):
            for i, item in enumerate(sketch_path):
                field = fields[i]
                path = item.get(field)
                tag = sketch_tag[i].get(field)

                if tag == 'clipasso':
                    tmp_data = get_sketch_data(DATA_DIR / path, pattern="*.svg")
                else:
                    tmp_data = get_sketch_data(DATA_DIR / path, pattern="*.png")
                data_dict_list.append(tmp_data)
            # Merge all the dict in the list data_dict_list
            result = self.merge_dicts_custom(data_dict_list)
        else:
            result = get_sketch_data(DATA_DIR / self.hparams.sketch_path, pattern="*.png")

        return result

    @staticmethod
    def merge_dicts_custom(dict_list):
        merged = defaultdict(list)
        for d in dict_list:
            for k, v in d.items():
                merged[k].extend(v)
        return merged

    def __getitem__(self, idx):
        new_idx = idx
        if self.repeat is not None and self.repeat > 1:
            new_idx = idx // self.repeat

        mesh_name, spa_idx = self.get_mesh_name(new_idx)

        if mesh_name:
            sketch_paths = self.sketch_data[mesh_name]
            if not sketch_paths:
                # logger.error(f"No sketch found for {mesh_name}")
                mesh_name = random.choice(self.sketch_keys)
                sketch_paths = self.sketch_data[mesh_name]
        else:
            keys = list(self.sketch_data.keys())
            mesh_name = random.choice(keys)
            sketch_paths = self.sketch_data[mesh_name]

        if self.repeat and self.repeat > 1:
            sketch_idx = int(idx % img_num)
        else:
            # Get a random number from 0 to 6
            sketch_idx = np.random.randint(0, img_num)

        sketch_path = sketch_paths[sketch_idx]

        if sketch_path.endswith(".svg"):
            svg = SVG(sketch_path)
            img = svg_to_img(svg)
        else:
            img = Image.open(sketch_path).convert('RGB')
        img = self.transform_train(img)  # img.shape = [3, 256, 256]

        latents = []
        for k in self.hparams.data_keys:
            data = torch.from_numpy(self.data[k][spa_idx])
            latents.append(data)

        item = latents + [img]
        if self.hparams.get("concat_data"):
            latents = torch.cat(latents, -1)
            return latents, img

        return item


class DatasetBuilder(object):
    def __init__(self, hparams):
        self.hparams = hparams

    def build_dataset(self, stage):
        if hasattr(self, f"data_{stage}"):
            return getattr(self, f"data_{stage}")

        if self.hparams.dataset_kwargs.sketch_tag == 'clipasso':
            ds_class = (
                CLIPassoSketchDataset
            )
        elif self.hparams.dataset_kwargs.sketch_tag == 'informative_drawings':
            ds_class = (
                InformativeSketchDataset
            )
        else:
            # ds_class = (
            #     ShapNetDataset
            # )
            ds_class = (
                AllDataset
            )
            logger.error(f"Unknown sketch_tag: {self.hparams.dataset_kwargs.sketch_tag}")
        if stage == "train":
            ds = ds_class(**self.hparams.dataset_kwargs)
        else:
            dataset_kwargs = self.hparams.dataset_kwargs.copy()
            dataset_kwargs["repeat"] = 1
            ds = ds_class(**dataset_kwargs)
        setattr(self, f"data_{stage}", ds)
        return ds

    def build_dataloader(self, stage):
        ds = self.build_dataset(stage)

        return torch.utils.data.DataLoader(
            ds,
            batch_size=self.hparams.batch_size,
            shuffle=stage == "train",
            drop_last=stage == "train",
            num_workers=self.hparams.num_workers,
            pin_memory=True,
            persistent_workers=True,
        )

    def train_dataloader(self):
        return self.build_dataloader("train")

    def val_dataloader(self):
        return self.build_dataloader("val")

    def test_dataloader(self):
        return self.build_dataloader("test")


if __name__ == "__main__":
    config_3d = OmegaConf.load("configs/data_3d/chair_salad.yaml")
    # config_sketch = OmegaConf.load("configs/data_sketch/informative_drawings_sketch_anime_style.yaml")
    config_sketch = OmegaConf.load("configs/data_sketch/all_sketch.yaml")
    # config_test = OmegaConf.load("configs/data_test/shapenet.yaml")
    config = OmegaConf.merge(config_3d, config_sketch)
    # dataset = CLIPassoSketchDataset(data_dir)
    # dataset = InformativeSketchDataset(config.latent_path, **config)
    dataset = AllDataset(config.latent_path, **config)
    for i in range(len(dataset)):
        print(i)
        print(dataset[i])
