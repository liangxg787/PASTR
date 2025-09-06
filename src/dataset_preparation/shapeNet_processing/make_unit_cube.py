# -*- coding: UTF-8 -*-
"""
@Time : 15/06/2025 10:25
@Author : Xiaoguang Liang
@File : make_unit_cube.py
@Project : Enhancing_Sketch-to-3D_Controllability
"""
import os
from tqdm import tqdm

from spaghetti.utils.mesh_utils import to_unit_cube
from spaghetti.utils import files_utils
from configs.log_config import logger


def process_all_meshes(mesh_path, output_path):
    for obj_folder in tqdm(os.listdir(mesh_path)):
        obj_path_name = os.path.join(mesh_path, obj_folder)
        if os.path.isdir(obj_path_name):
            for sub_obj_path in os.listdir(obj_path_name):
                if sub_obj_path == 'models':
                    sub_obj_path_name = os.path.join(obj_path_name, sub_obj_path)
                    output_path_tmp = ''.join([output_path, '/', obj_folder])
                    if not os.path.exists(output_path_tmp):
                        os.makedirs(output_path_tmp)

                    for obj_file in os.listdir(sub_obj_path_name):
                        if obj_file.endswith('.obj'):
                            obj_file_path = os.path.join(sub_obj_path_name, obj_file)

                            mesh = files_utils.load_mesh(obj_file_path)
                            meshes, (center, scale) = to_unit_cube(mesh)

                            file_name = obj_file.split('.')[0]
                            save_file = f"{output_path_tmp}/{file_name}"
                            files_utils.export_mesh(meshes, save_file)
                            logger.info(f'center: {center}')
                            logger.info(f'scale: {scale}')


if __name__ == '__main__':
    mesh_dir = 'data/shapenet/'
    output_dir = 'output/unit_cube'
    process_all_meshes(mesh_dir, output_dir)
