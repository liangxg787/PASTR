# -*- coding: UTF-8 -*-
"""
@Time : 19/07/2025 16:03
@Author : Xiaoguang Liang
@File : make_amateur_3d.py
@Project : Enhancing_Sketch-to-3D_Controllability
"""
import os
import shutil

from tqdm import tqdm

from src.utils.data_util import get_sketch_data
from configs.global_setting import DATA_DIR
from configs.log_config import logger


def get_amateur_3d_data(sketch_path, obj_path, amateur_3d_path):
    sketch_data = get_sketch_data(sketch_path, pattern="*.svg")
    mesh_names = list(sketch_data.keys())

    logger.info(f"The amount of sketch files: {len(mesh_names)}")

    if not os.path.exists(amateur_3d_path):
        os.mkdir(amateur_3d_path)

    for sub_folder in tqdm(os.listdir(obj_path)):
        if sub_folder in mesh_names:
            sub_folder_path = os.path.join(obj_path, sub_folder)
            for obj_file in os.listdir(sub_folder_path):
                if obj_file.endswith('.obj') and 'manifold' not in obj_file:
                    new_obj_file = os.path.join(amateur_3d_path, f"{sub_folder}.obj")
                    shutil.copy(os.path.join(sub_folder_path, obj_file), new_obj_file)
            # if os.path.isdir(sub_folder_path):  # Check if it's a directory
            #     shutil.copytree(sub_folder_path, os.path.join(amateur_3d_path,
            #                                                   sub_folder))  # Use copytree for directories
            # else:
            #     shutil.copy(sub_folder_path, amateur_3d_path)  # Use copy for files


if __name__ == '__main__':
    sketch_dir = DATA_DIR / 'AmateurSketch3D/chair_svg'
    obj_dir = DATA_DIR / "03001627"
    amateur_3d_dir = DATA_DIR / 'AmateurSketch3D/chair_3d'
    get_amateur_3d_data(sketch_dir, obj_dir, amateur_3d_dir)
