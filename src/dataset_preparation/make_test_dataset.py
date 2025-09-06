# -*- coding: UTF-8 -*-
"""
@Time : 06/07/2025 18:44
@Author : Xiaoguang Liang
@File : make_test_dataset.py
@Project : Enhancing_Sketch-to-3D_Controllability
"""
import os
import json
import shutil
import random

import numpy as np
from tqdm import tqdm

from configs.global_setting import DATA_DIR
from src.utils.data_util import get_sketch_data


def filter_mesh_names(salad_ids_path, shapenet_folder, test_dataset_folder):
    with open(salad_ids_path, "r") as fp:
        mesh_ids = json.load(fp)
    mesh_names = [v[0] for _, v in mesh_ids.items()]
    mesh_names = list(set(mesh_names))

    if not os.path.exists(test_dataset_folder):
        os.makedirs(test_dataset_folder)

    # Get all the mesh names from 03001627 (Chair)
    for sub_folder in tqdm(os.listdir(shapenet_folder)):
        if sub_folder not in mesh_names:
            sub_folder_path = os.path.join(shapenet_folder, sub_folder)
            if os.path.isdir(sub_folder_path):  # Check if it's a directory
                shutil.copytree(sub_folder_path, os.path.join(test_dataset_folder,
                                                              sub_folder))  # Use copytree for directories
            else:
                shutil.copy(sub_folder_path, test_dataset_folder)  # Use copy for files


def sample_shapes(shapenet_folder, test_dataset_folder, num_samples=2000):
    if not os.path.exists(test_dataset_folder):
        os.makedirs(test_dataset_folder)

    # Make 2000 random numbers between 0 and 6778
    mesh_names = random.sample(range(6778), num_samples)
    # Get all the mesh names from 03001627 (Chair)
    for i, sub_folder in tqdm(enumerate(os.listdir(shapenet_folder))):
        if i in mesh_names:
            sub_folder_path = os.path.join(shapenet_folder, sub_folder)
            if os.path.isdir(sub_folder_path):  # Check if it's a directory
                shutil.copytree(sub_folder_path, os.path.join(test_dataset_folder,
                                                              sub_folder))  # Use copytree for directories
            else:
                shutil.copy(sub_folder_path, test_dataset_folder)  # Use copy for files


def get_mesh_names_from_dataset(chairs_list):
    with open(chairs_list, 'r') as f:
        mesh_names = [line.strip() for line in f.readlines()]

    return mesh_names


def sample_500_shapes(sample_meshes, shapenet_folder, test_dataset_folder, sketch_folder, test_sketch_folder):
    # Get all the mesh names from 03001627 (Chair)
    if not os.path.exists(test_dataset_folder):
        os.makedirs(test_dataset_folder)
        for sub_folder in tqdm(os.listdir(shapenet_folder)):
            if sub_folder in sample_meshes:
                sub_folder_path = os.path.join(shapenet_folder, sub_folder)
                if os.path.isdir(sub_folder_path):  # Check if it's a directory
                    shutil.copytree(sub_folder_path, os.path.join(test_dataset_folder,
                                                                  sub_folder))  # Use copytree for directories
                else:
                    shutil.copy(sub_folder_path, test_dataset_folder)  # Use copy for files

    if 'clipasso' in str(sketch_folder):
        tmp_data = get_sketch_data(sketch_folder, pattern="*.svg")
    else:
        tmp_data = get_sketch_data(sketch_folder, pattern="*.png")

    for i, mesh_n in tqdm(enumerate(sample_meshes)):
        sketch_path = tmp_data.get(mesh_n, None)
        if sketch_path:
            # sketch_idx = np.random.randint(0, 6)
            for j in range(6):
                sketch_idx = j
                one_sketch_path = sketch_path[sketch_idx]
                if 'view_60' in one_sketch_path:
                    break
            if i == 0:
                sub_sketch_path = one_sketch_path.split('/')[-2]
                new_sketch_folder = os.path.join(test_sketch_folder, sub_sketch_path)
                if not os.path.exists(new_sketch_folder):
                    os.makedirs(new_sketch_folder)
            shutil.copy(one_sketch_path, new_sketch_folder)
        else:
            raise ValueError(f"{mesh_n} not found")


def make_all_test_dataset(chairs_list, shapenet_folder, test_dataset_folder, test_sketch_folder):
    sketch_f_1 = DATA_DIR / "informative_drawings_sketches/anime_style"
    sketch_f_2 = DATA_DIR / "informative_drawings_sketches/opensketch_style"
    sketch_f_3 = DATA_DIR / "clipasso_16strokes"
    sketch_f_4 = DATA_DIR / "clipasso_32strokes"
    sketch_paths = [sketch_f_1, sketch_f_2, sketch_f_3, sketch_f_4]
    mesh_name = get_mesh_names_from_dataset(chairs_list)
    sample_meshes = random.sample(mesh_name, 1000)

    for sketch_path in tqdm(sketch_paths):
        sample_500_shapes(sample_meshes, shapenet_folder, test_dataset_folder, sketch_path, test_sketch_folder)


if __name__ == "__main__":
    # salad_ids_path = DATA_DIR / "salad_data/ids.json"
    # shapenet_folder = DATA_DIR / "03001627"
    shapenet_folder = "/nobackup/babbage/users/xl01339/03001627"
    test_dataset_folder = DATA_DIR / "test_dataset_03001627"
    # # filter_mesh_names(salad_ids_path, shapenet_folder, test_dataset_folder)
    # sample_shapes(shapenet_folder, test_dataset_folder)
    chairs_list = DATA_DIR / "chairs_list.txt"

    test_sketch_folder = DATA_DIR / "test_sketch_03001627"

    make_all_test_dataset(chairs_list, shapenet_folder, test_dataset_folder, test_sketch_folder)
