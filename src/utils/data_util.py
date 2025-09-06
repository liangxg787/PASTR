# -*- coding: UTF-8 -*-
"""
@Time : 07/07/2025 21:42
@Author : Xiaoguang Liang
@File : data_util.py
@Project : Enhancing_Sketch-to-3D_Controllability
"""
import gc
import os
import sys
import re
import random
from typing import Dict, Any, Callable, Iterator, Union
from fnmatch import fnmatch
from collections import defaultdict
import importlib
import itertools as itt

import torch
import numpy as np
import cv2

from configs.log_config import logger
from configs.global_setting import DATA_DIR

SEP = '_'
SEP_ = '-'


def get_sketch_data(sketch_dir, pattern="*best.svg"):
    if not os.path.exists(sketch_dir):
        raise FileNotFoundError(f"{sketch_dir} does not exist!")
    data = defaultdict(list)
    for path, subdirs, files in os.walk(sketch_dir):
        for name in files:
            if SEP in name:
                shape_name = name.split(SEP)[0]
            elif SEP_ in name:
                shape_name = name.split(SEP_)[0]
            else:
                raise ValueError(f"{name} does not contain segment symbol {SEP} or {SEP_}")
            if fnmatch(name, pattern):
                file_path = os.path.join(path, name)
                data[shape_name].append(file_path)

    logger.info(f"The amount of sketch files: {len(data)}")
    return data


def get_amateur_sketch_data(sketch_dir, pattern="*.png"):
    if not os.path.exists(sketch_dir):
        raise FileNotFoundError(f"{sketch_dir} does not exist!")
    data = defaultdict(list)
    for path, subdirs, files in os.walk(sketch_dir):
        for name in files:
            if fnmatch(name, pattern):
                file_path = os.path.join(path, name)
                shape_name = path.split("/")[-1]
                data[shape_name].append(file_path)

    logger.info(f"The amount of sketch files: {len(data)}")
    return data


def get_data_dict(data_path):
    result = defaultdict()
    for folder in os.listdir(data_path):
        path = os.path.join(data_path, folder)
        result[folder] = path
    return result


def get_test_dataset(test_dataset_dir, pattern="*.obj"):
    if not os.path.exists(test_dataset_dir):
        raise FileNotFoundError(f"{test_dataset_dir} does not exist")
    data = defaultdict()
    for path, subdirs, files in os.walk(test_dataset_dir):
        for name in files:
            if fnmatch(name, pattern):
                if len(name) < 30:
                    if 'model_normalized' in name:
                        shape_name = path.split("/")[-2]
                    else:
                        shape_name = path.split(".")[-1]
                else:
                    shape_name = name.split(".")[0]
                file_path = os.path.join(path, name)
                data[shape_name] = file_path

    logger.info(f"The amount of test dataset: {len(data)}")
    return data


def get_test_images(test_sketch_path):
    test_sketch = get_sketch_data(test_sketch_path,
                                  pattern="*.jpg")
    sample_mesh_names = down_sample(test_sketch, 3)
    test_images = list(map(lambda x: test_sketch[x][0], sample_mesh_names))
    del test_sketch
    del sample_mesh_names
    gc.collect()
    return test_images


def down_sample(objs: Dict, num_samples=300):
    mesh_names = list(objs.keys())
    sample_meshes = random.sample(mesh_names, num_samples)
    return sample_meshes


def make_seed(*args):
    seed = 0
    for arg in args:
        seed = (seed * 31 + hash(arg)) & 0x7FFFFFFF
    return seed


class PipelineStage:
    def invoke(self, *args, **kw):
        raise NotImplementedError


def identity(x: Any) -> Any:
    """Return the argument as is."""
    return x


def safe_eval(s: str, expr: str = "{}"):
    """Evaluate the given expression more safely."""
    if re.sub("[^A-Za-z0-9_]", "", s) != s:
        raise ValueError(f"safe_eval: illegal characters in: '{s}'")
    return eval(expr.format(s))


def lookup_sym(sym: str, modules: list):
    """Look up a symbol in a list of modules."""
    for mname in modules:
        module = importlib.import_module(mname, package="webdataset")
        result = getattr(module, sym, None)
        if result is not None:
            return result
    return None


def repeatedly0(
        loader: Iterator, nepochs: int = sys.maxsize, nbatches: int = sys.maxsize
):
    """Repeatedly returns batches from a DataLoader."""
    for _ in range(nepochs):
        yield from itt.islice(loader, nbatches)


def guess_batchsize(batch: Union[tuple, list]):
    """Guess the batch size by looking at the length of the first element in a tuple."""
    return len(batch[0])


def repeatedly(
        source: Iterator,
        nepochs: int = None,
        nbatches: int = None,
        nsamples: int = None,
        batchsize: Callable[..., int] = guess_batchsize,
):
    """Repeatedly yield samples from an iterator."""
    epoch = 0
    batch = 0
    total = 0
    while True:
        for sample in source:
            yield sample
            batch += 1
            if nbatches is not None and batch >= nbatches:
                return
            if nsamples is not None:
                total += guess_batchsize(sample)
                if total >= nsamples:
                    return
        epoch += 1
        if nepochs is not None and epoch >= nepochs:
            return


def pytorch_worker_info(group=None):  # sourcery skip: use-contextlib-suppress
    """Return node and worker info for PyTorch and some distributed environments."""
    rank = 0
    world_size = 1
    worker = 0
    num_workers = 1
    if "RANK" in os.environ and "WORLD_SIZE" in os.environ:
        rank = int(os.environ["RANK"])
        world_size = int(os.environ["WORLD_SIZE"])
    else:
        try:
            import torch.distributed

            if torch.distributed.is_available() and torch.distributed.is_initialized():
                group = group or torch.distributed.group.WORLD
                rank = torch.distributed.get_rank(group=group)
                world_size = torch.distributed.get_world_size(group=group)
        except ModuleNotFoundError:
            pass
    if "WORKER" in os.environ and "NUM_WORKERS" in os.environ:
        worker = int(os.environ["WORKER"])
        num_workers = int(os.environ["NUM_WORKERS"])
    else:
        try:
            import torch.utils.data

            worker_info = torch.utils.data.get_worker_info()
            if worker_info is not None:
                worker = worker_info.id
                num_workers = worker_info.num_workers
        except ModuleNotFoundError:
            pass

    return rank, world_size, worker, num_workers


def pytorch_worker_seed(group=None):
    """Compute a distinct, deterministic RNG seed for each worker and node."""
    rank, world_size, worker, num_workers = pytorch_worker_info(group=group)
    return rank * 1000 + worker


def worker_init_fn(_):
    worker_info = torch.utils.data.get_worker_info()
    worker_id = worker_info.id

    # dataset = worker_info.dataset
    # split_size = dataset.num_records // worker_info.num_workers
    # # reset num_records to the true number to retain reliable length information
    # dataset.sample_ids = dataset.valid_ids[worker_id * split_size:(worker_id + 1) * split_size]
    # current_id = np.random.choice(len(np.random.get_state()[1]), 1)
    # return np.random.seed(np.random.get_state()[1][current_id] + worker_id)

    return np.random.seed(np.random.get_state()[1][0] + worker_id)


def collation_fn(samples, combine_tensors=True, combine_scalars=True):
    """

    Args:
        samples (list[dict]):
        combine_tensors:
        combine_scalars:

    Returns:

    """

    result = {}

    keys = samples[0].keys()

    for key in keys:
        result[key] = []

    for sample in samples:
        for key in keys:
            val = sample[key]
            result[key].append(val)

    for key in keys:
        val_list = result[key]
        if isinstance(val_list[0], (int, float)):
            if combine_scalars:
                result[key] = np.array(result[key])

        elif isinstance(val_list[0], torch.Tensor):
            if combine_tensors:
                result[key] = torch.stack(val_list)

        elif isinstance(val_list[0], np.ndarray):
            if combine_tensors:
                result[key] = np.stack(val_list)

    return result


def recenter(image, border_ratio: float = 0.2):
    """ recenter an image to leave some empty space at the image border.

    Args:
        image (ndarray): input image, float/uint8 [H, W, 3/4]
        mask (ndarray): alpha mask, bool [H, W]
        border_ratio (float, optional): border ratio, image will be resized to (1 - border_ratio). Defaults to 0.2.

    Returns:
        ndarray: output image, float/uint8 [H, W, 3/4]
    """

    if image.shape[-1] == 4:
        mask = image[..., 3]
    else:
        mask = np.ones_like(image[..., 0:1]) * 255
        image = np.concatenate([image, mask], axis=-1)
        mask = mask[..., 0]

    H, W, C = image.shape

    size = max(H, W)
    result = np.zeros((size, size, C), dtype=np.uint8)

    coords = np.nonzero(mask)
    x_min, x_max = coords[0].min(), coords[0].max()
    y_min, y_max = coords[1].min(), coords[1].max()
    h = x_max - x_min
    w = y_max - y_min
    if h == 0 or w == 0:
        raise ValueError('input image is empty')
    desired_size = int(size * (1 - border_ratio))
    scale = desired_size / max(h, w)
    h2 = int(h * scale)
    w2 = int(w * scale)
    x2_min = (size - h2) // 2
    x2_max = x2_min + h2

    y2_min = (size - w2) // 2
    y2_max = y2_min + w2

    result[x2_min:x2_max, y2_min:y2_max] = cv2.resize(image[x_min:x_max, y_min:y_max], (w2, h2),
                                                      interpolation=cv2.INTER_AREA)

    bg = np.ones((result.shape[0], result.shape[1], 3), dtype=np.uint8) * 255

    mask = result[..., 3:].astype(np.float32) / 255
    result = result[..., :3] * mask + bg * (1 - mask)

    mask = mask * 255
    result = result.clip(0, 255).astype(np.uint8)
    mask = mask.clip(0, 255).astype(np.uint8)
    return result, mask


def get_mesh_names_from_dataset(chairs_list):
    with open(chairs_list, 'r') as f:
        mesh_names = [line.strip() for line in f.readlines()]

    return mesh_names