# -*- coding: UTF-8 -*-
"""
@Time : 25/08/2025 11:11
@Author : Xiaoguang Liang
@File : render_util.py
@Project : Enhancing_Sketch-to-3D_Controllability
"""
import os

import numpy as np
from spaghetti.utils.files_utils import load_mesh, save_image

from src.utils.visual_util import render_mesh
from configs.global_setting import BASE_DIR
from src.utils.common import timer


def get_rendered_mesh(mesh_path, mesh_name, output_path):
    v, f = load_mesh(str(mesh_path))

    if not os.path.exists(output_path):
        os.makedirs(output_path)

    # camPos = np.array([2, 2, -2])
    # image = render_mesh(v, f, resolution=(1024, 1024), samples=512, camPos=camPos)
    image = render_mesh(v, f, resolution=(1024, 1024), samples=512)

    save_path = output_path / f'{mesh_name}.png'
    save_image(image, str(save_path))


if __name__ == '__main__':
    # file_path = BASE_DIR / 'output/model_normalized.obj'
    # file_path = BASE_DIR / 'output/1a6f615e8b1b5ae4dbbc9440457e303e_manifold.obj'
    file_path = BASE_DIR / 'output/1a6f615e8b1b5ae4dbbc9440457e303e_manifold_plus.obj'

    output_path = BASE_DIR / 'output/rendered_meshes'
    mesh_name = file_path.stem

    with timer('Render mesh'):
        get_rendered_mesh(file_path, mesh_name,  output_path)
