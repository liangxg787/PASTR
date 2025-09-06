# -*- coding: UTF-8 -*-
"""
@Time : 09/06/2025 10:32
@Author : Xiaoguang Liang
@File : convert_to_manifold.py
@Project : Enhancing_Sketch-to-3D_Controllability
"""
import os
from fnmatch import fnmatch
import multiprocessing as mp
import traceback

from tqdm import tqdm
from joblib import Parallel, delayed

from configs.log_config import logger
from src.utils.common import timer

manifold_path = 'external/Manifold/build/manifold'
manifoldPlus_path = 'external/ManifoldPlus/build/manifold'


def to_manifold(args):
    path, name = args
    try:
        os.system(
            f"{manifold_path} {os.path.join(path, 'model.obj')} {os.path.join(path, 'model_manifold.obj')}")
        print(f"Done manifold conversion to {os.path.join(path, 'model.obj')}!")
    except:
        traceback.print_exc()


def to_manifold_plus(args):
    path, name = args
    try:
        os.system(
            f"{manifoldPlus_path} --input {os.path.join(path, 'model.obj')} --output {os.path.join(path, 'model_manifold_plus.obj')} --depth 8")
        print(f"Done manifold_plus conversion to {os.path.join(path, 'model.obj')}!")
    except:
        traceback.print_exc()


def parallel_to_manifold(data_path, pattern="*.obj", cpu_num=10, plus=True):
    args = []
    for path, subdirs, files in os.walk(data_path):
        for name in files:
            if fnmatch(name,
                       pattern) and not "manifold" in name and (not os.path.exists(
                os.path.join(path, 'model_manifold.obj')) or not os.path.exists(
                os.path.join(path, 'model_manifold_puls.obj'))):
                args.append((path, name))
                # print(f"added path {path} {name}")

    logger.info(f"The amount of files: {len(args)}")

    # workers = 12
    # pool = mp.Pool(workers)
    # pool.map(to_manifold, args)

    if plus:
        Parallel(n_jobs=cpu_num, backend='loky', verbose=10)(
            delayed(to_manifold_plus)(batch) for batch in tqdm(args))
    else:
        Parallel(n_jobs=cpu_num, backend='loky', verbose=10)(
            delayed(to_manifold)(batch) for batch in tqdm(args))


if __name__ == '__main__':
    # root = 'dataset/03001627_10'
    root = 'dataset/03001627'
    # pattern = "*.obj"

    with timer("All tasks"):
        parallel_to_manifold(root)
