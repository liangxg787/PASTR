# -*- coding: UTF-8 -*-
"""
@Time : 19/08/2025 17:06
@Author : Xiaoguang Liang
@File : make_latents_dataset.py
@Project : SENS
"""
import re
from fnmatch import fnmatch
from collections import defaultdict

import torch
import numpy as np
from h5py import File
from tqdm import trange

from spaghetti import constants
from spaghetti.utils import files_utils

from src.utils.spaghetti_util import load_spaghetti
from configs.global_setting import device


@torch.no_grad()
def flatten_gmms_item(x):
    """
    Input: [B,1,G,*shapes]
    Output: [B,G,-1]
    """
    return x.reshape(x.shape[0], x.shape[2], -1)


@torch.no_grad()
def batch_gmms_to_gaus(gmms):
    """
    Input:
        [T(B,1,G,3), T(B,1,G,3,3), T(B,1,G), T(B,1,G,3)]
    Output:
        T(B,G,16)
    """
    if isinstance(gmms[0], list):
        gaus = gmms[0].copy()
    else:
        gaus = list(gmms).copy()

    gaus = [flatten_gmms_item(x) for x in gaus]
    return torch.cat(gaus, -1)


def get_mesh_names_from_dataset(chairs_list):
    with open(chairs_list, 'r') as f:
        mesh_names = [line.strip() for line in f.readlines()]

    return mesh_names


# opt_ = Options(tag='chairs_sym_hard').load()
# spaghetti = occ_inference.Inference(opt_)


data_name = 'chairs_large'
spaghetti = load_spaghetti(device=device, tag=data_name)


def inference():
    # opt_ = Options(device=CUDA(0), tag='chairs', model_name='occ_gmm').load()
    tag = 'chairs_model'
    weight_cls = "1.0"
    # opt_ = SketchOptions(tag=tag, weight_cls=weight_cls)
    # opt_ = Options(device=CUDA(0), tag='chairs', model_name='occ_gmm', dataset_size=6755).load()

    chairs_list = "assets/data/dataset_chair_preprocess/chairs_list.txt"
    names = get_mesh_names_from_dataset(chairs_list)
    # spaghetti.plot_('occ_gmm', names=names)
    # spaghetti.plot('occ_gmm', names=names)

    out_root = f'{constants.DATA_ROOT}/dataset_chair_preprocess/'
    zh = torch.from_numpy(files_utils.load_np(f'{out_root}zh_0'))
    print(len(zh))

    all_zh_base = []
    all_z_gmm = []
    mesh_names = []

    folder_name = 'occ_gmm'
    for i in trange(len(zh)):
        zh_ = zh[i]
        zh_ = zh_.to(device)
        out_z, gmms = spaghetti.model.occ_former.forward_mid(zh_.unsqueeze(0).unsqueeze(0))

        # z = torch.tensor(out_z[0], device=device)
        gmms = [x.to(device) for x in gmms[0]]

        # zh_out, _ = spaghetti.model.merge_zh(z, gmms)

        # mesh = spaghetti.get_mesh(zh_out, 256, None)
        # output_path = 'demo.obj'
        # files_utils.export_mesh(mesh, output_path)

        # break

        shape_gmm = batch_gmms_to_gaus(gmms)
        shape_gmm = shape_gmm.squeeze(0).detach().cpu()

        all_zh_base.append([x.detach().cpu() for x in out_z[0]])
        all_z_gmm.append(shape_gmm)
        mesh_names.append(names[i])

        # files_utils.save_pickle(out_z.detach().cpu(),
        #                         f'{opt_.cp_folder}/{folder_name}/{names[i]}')
        # files_utils.export_gmm(gmms[0], i, f'{opt_.cp_folder}/{folder_name}/{names[i]}')

        # print(len(out_z))

    all_zh_base = np.array(all_zh_base)
    all_z_gmm = np.array(all_z_gmm)
    # Calculate global men for gmms
    z_gmm_mean = np.mean(all_z_gmm, axis=(0, 1))
    print(f'z_gmm_mean shape: {z_gmm_mean.shape}')
    # Calculate global standard deviation for gmms
    z_gmm_std = np.std(all_z_gmm, axis=(0, 1))
    print(f'z_gmm_std shape: {z_gmm_std.shape}')
    # Save data
    print('Save all data ...')
    save_file_name = 'spaghetti_chair_latents_6755_samples_mean_std'
    data = {'s_j_affine': all_zh_base, 'g_js_affine': all_z_gmm, 'mean': z_gmm_mean, 'std': z_gmm_std,
            'mesh_names': mesh_names}

    output_name = 'assets'
    with File(f"{output_name}/{save_file_name}.hdf5", "w") as f:
        for k, v in data.items():
            f[k] = v


if __name__ == '__main__':
    inference()
