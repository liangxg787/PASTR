# -*- coding: UTF-8 -*-
"""
@Time : 09/06/2025 11:45
@Author : Xiaoguang Liang
@File : make_spaghetti_representation.py
@Project : Enhancing_Sketch-to-3D_Controllability
"""
import os
import shutil
import numpy as np
from pathlib import Path
from fnmatch import fnmatch
from functools import partial

import torch
from h5py import File
from tqdm import tqdm
from spaghetti import options
from spaghetti.shape_inversion import MeshProjectionMid
from sentry_sdk import capture_exception
from joblib import Parallel, delayed

from configs.global_setting import BASE_DIR, device
from src.utils.spaghetti_util import add_spaghetti_path, delete_spaghetti_path
from src.utils.common import timer
from configs.log_config import logger

EPOCHS = 500
cpu_num = 6


def merge_zh_step_a(gmms):
    b, gp, g, _ = gmms[0].shape
    mu, p, phi, eigen = [item.view(b, gp * g, *item.shape[3:]) for item in gmms]
    p = p.reshape(*p.shape[:2], -1)
    # z_gmm.shape = (1, 16, 16)
    z_gmm = torch.cat((mu, p, phi.unsqueeze(-1), eigen), dim=2).detach()
    # z_gmm = self.from_gmm(z_gmm) # z_gmm.shape = (1, 16, 512)
    return z_gmm


def make_representation_for_one(obj_file_path, model_name='chairs_large', mesh_name='', output_path='',
                                num_epochs=EPOCHS, plot=True, save_data=True):
    opt = options.Options(tag=model_name)
    output_path = str(output_path)

    if not os.path.exists(obj_file_path):
        raise FileNotFoundError(f'{obj_file_path} not found!')
    obj_file_path = str(obj_file_path)

    model = MeshProjectionMid(opt, obj_file_path, mesh_name)

    # predict with spaghetti
    for i in range(num_epochs // 2):
        if model.early_stop(model.train_epoch(i), i):
            break
    model.switch_embedding()
    for i in range(num_epochs):
        if model.early_stop(model.train_epoch(i), i):
            break

    z_d = model.mid_embeddings(
        torch.zeros(1, device=model.device, dtype=torch.int64)
    ).view(1, model.opt.num_gaussians, -1)
    zh_base, gmms = model.model.decomposition_control.forward_mid(z_d)
    z_gmm = merge_zh_step_a(gmms)

    if plot:
        mesh_name_ = output_path.split('/')[-1] + f'/{mesh_name}'
        z_gmm_ = model.model.from_gmm(z_gmm)
        zh_ = zh_base + z_gmm_
        zh_, attn = model.model.mixing_network.forward_with_attention(zh_, mask=None)
        numbers = model.get_new_ids(mesh_name, 1)
        model.plot_occ(zh_, zh_base, gmms, numbers, mesh_name_, verbose=False, res=256)

    zh_base = zh_base.cpu().detach().numpy()
    zh_base = zh_base.reshape(16, 512)
    z_gmm = z_gmm.cpu().detach().numpy()
    z_gmm = z_gmm.reshape(16, 16)

    # Save data
    if save_data:
        logger.info('Save all data ...')
        data = {'s_j_affine': zh_base, 'g_js_affine': z_gmm,
                'mesh_names': mesh_name}
        with File(f"{output_path}/{mesh_name}.hdf5", "w") as f:
            for k, v in data.items():
                f[k] = v

    return zh_base, z_gmm


def make_representation_by_inference(obj_file_path, model_name='chairs_large', mesh_name='', output_path='',
                                     num_epochs=EPOCHS, plot=True, save_data=True):
    pass


def make_representation_by_inversion(obj_file_path, output_name='', model_name='chairs_large',
                                     num_epochs=EPOCHS):
    opt = options.Options(tag=model_name, decomposition_network='mlp')
    # if not os.path.exists(output_name):
    #     os.makedirs(output_name)
    if not os.path.exists(obj_file_path):
        raise FileNotFoundError(f'{obj_file_path} not found!')
    obj_file_path = str(obj_file_path)

    model = MeshProjectionMid(opt, obj_file_path, output_name)
    model.invert(num_epochs)


def make_representation(model_name, output_path, manifold_files_path):
    sub_folders = os.listdir(manifold_files_path)
    add_spaghetti_path()
    all_zh_base = []
    all_z_gmm = []
    mesh_names = []

    for folder in tqdm(sub_folders):
        sub_path = os.path.join(manifold_files_path, folder)
        if os.path.isdir(sub_path):
            files = list(Path(sub_path).glob('*manifold.obj'))
            if not files:
                raise FileNotFoundError(f'manifold.obj is not found in {sub_path}!')
            manifold_file = str(files[0])
            logger.info(f'Processing {manifold_file} ...')
            if not os.path.exists(output_path):
                os.makedirs(output_path)
            zh_base, z_gmm = make_representation_for_one(manifold_file, model_name=model_name,
                                                         mesh_name=folder, output_path=output_path,
                                                         save_data=False)

            all_zh_base.append(zh_base)
            all_z_gmm.append(z_gmm)
            mesh_names.append(folder)

    delete_spaghetti_path()
    all_zh_base = np.array(all_zh_base)
    all_z_gmm = np.array(all_z_gmm)
    # Calculate global men for gmms
    z_gmm_mean = np.mean(all_z_gmm, axis=(0, 1))
    logger.info(f'z_gmm_mean shape: {z_gmm_mean.shape}')
    # Calculate global standard deviation for gmms
    z_gmm_std = np.std(all_z_gmm, axis=(0, 1))
    logger.info(f'z_gmm_std shape: {z_gmm_std.shape}')
    # Save data
    logger.info('Save all data ...')
    save_file_name = 'spaghetti_chair_latents_10_samples_mean_std'
    data = {'s_j_affine': all_zh_base, 'g_js_affine': all_z_gmm, 'mean': z_gmm_mean, 'std': z_gmm_std,
            'mesh_names': mesh_names}
    with File(f"{output_name}/{save_file_name}.hdf5", "w") as f:
        for k, v in data.items():
            f[k] = v


def make_one_representation_for_parallel(manifold_mash_name, model_name='chairs_large',
                                         output_path='',
                                         num_epochs=EPOCHS):
    obj_file_path, mesh_name = manifold_mash_name
    if not os.path.exists(output_path):
        os.makedirs(output_path)

    _, _ = make_representation_for_one(obj_file_path, model_name=model_name,
                                       mesh_name=mesh_name, num_epochs=num_epochs)


def parallel_make_representation(model_name, output_path, manifold_files_path):
    sub_folders = os.listdir(manifold_files_path)
    add_spaghetti_path()
    all_manifold_mash_name = []
    for folder in tqdm(sub_folders):
        sub_path = os.path.join(manifold_files_path, folder)
        if os.path.isdir(sub_path):
            files = list(Path(sub_path).glob('*manifold.obj'))
            if not files:
                raise FileNotFoundError(f'manifold.obj is not found in {sub_path}!')
            manifold_file = str(files[0])
            all_manifold_mash_name.append((manifold_file, folder))

    _make_one_representation_for_parallel = partial(make_one_representation_for_parallel,
                                                    model_name=model_name,
                                                    output_path=output_path)
    Parallel(n_jobs=cpu_num, backend='multiprocessing', verbose=10)(
        delayed(_make_one_representation_for_parallel)(batch) for batch in tqdm(all_manifold_mash_name))
    delete_spaghetti_path()


def split_obj_dataset(folder, split_amount=230):
    i = 0
    j = 1
    new_folder = f"{folder}_split"
    for path, subdirs, files in os.walk(folder):
        for name in files:
            if fnmatch(name, "*manifold.obj"):
                i += 1
                # Move the file in the new directory
                old_path = str(os.path.join(path, name))
                mesh_name = old_path.split('/')[-2]
                tmp_new_folder = f"{new_folder}/{j}/{mesh_name}"
                if not os.path.exists(tmp_new_folder):
                    os.makedirs(tmp_new_folder)

                if i <= split_amount * j:
                    shutil.copy(old_path, tmp_new_folder)
                if i == split_amount * j:
                    j += 1


if __name__ == '__main__':
    import argparse

    torch.multiprocessing.set_start_method('spawn')

    parser = argparse.ArgumentParser(description="Make spaghetti representation ...")

    parser.add_argument('--folder-num', type=int, default=1)

    folder_num = parser.parse_args().folder_num

    model_name = 'chairs_large'

    # obj_file_path = BASE_DIR / 'dataset/03001627_10/1a6f615e8b1b5ae4dbbc9440457e303e/model_manifold.obj'
    obj_file_path = BASE_DIR / 'dataset/03001627_10/1a6f615e8b1b5ae4dbbc9440457e303e/model_manifold_plus.obj'
    output_path = BASE_DIR / 'output/03001627_10'
    # output_path = BASE_DIR / 'output/03001627'
    add_spaghetti_path()
    with timer('all tasks'):
        if not os.path.exists(output_path):
            os.makedirs(output_path)
        mesh_name = "1a6f615e8b1b5ae4dbbc9440457e303e"

        make_representation_by_inversion(obj_file_path, mesh_name)
        # make_representation_for_one(obj_file_path, model_name=model_name,
        #                             mesh_name=mesh_name, output_path=output_path)

    # manifold_shapeNet = BASE_DIR / 'dataset/03001627_10'
    # manifold_shapeNet = BASE_DIR / 'dataset/03001627'

    # split_obj_dataset(manifold_shapeNet)

    # manifold_shapeNet = BASE_DIR / f'dataset/03001627_split/{folder_num}'

    # try:
    #     with timer('all tasks'):
    #         # make_representation(model_name, output_name, manifold_shapeNet)
    #         parallel_make_representation(model_name, output_name, manifold_shapeNet)
    # except Exception as exc:
    #     capture_exception(exc)
    #     logger.error(exc)
    #     raise exc
