# -*- coding: UTF-8 -*-
"""
@Time : 09/06/2025 10:33
@Author : Xiaoguang Liang
@File : scale_to_unit_sphere.py
@Project : Enhancing_Sketch-to-3D_Controllability
"""
import os
from fnmatch import fnmatch
import multiprocessing as mp
import traceback
from spaghetti.utils.mesh_utils import to_unit_sphere
from spaghetti.utils import files_utils

from configs.global_setting import BASE_DIR


def to_sphere(args):
    path, name = args
    try:
        mesh = files_utils.load_mesh(os.path.join(path, name))
        mesh = to_unit_sphere(mesh, scale=0.90)
        files_utils.export_mesh(mesh, os.path.join(path, 'model_manifold_sphere.obj'))
        print(f"Done manifold conversion to {os.path.join(path, 'model_flipped.obj')}!")
    except:
        traceback.print_exc()


if __name__ == '__main__':

    root = BASE_DIR / 'dataset/03001627_10'
    pattern = "*.obj"

    args = []
    for path, subdirs, files in os.walk(root):
        for name in files:
            if fnmatch(name,
                       pattern) and not "manifold" in name and not os.path.exists(
                os.path.join(path, 'model_manifold_sphere.obj')):
                args.append((path, name))
                # print(f"added path {path} {name}")

    print(f"len of the arrays {len(args)}")

    workers = 12
    pool = mp.Pool(workers)
    pool.map(to_sphere, args)
