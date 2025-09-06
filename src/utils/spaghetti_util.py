# -*- coding: UTF-8 -*-
"""
@Time : 16/06/2025 10:17
@Author : Xiaoguang Liang
@File : spaghetti_util.py
@Project : Enhancing_Sketch-to-3D_Controllability
"""
import sys
from typing import Union

import numpy as np
import torch
from spaghetti.utils.mcubes_meshing import MarchingCubesMeshing
from spaghetti.custom_types import *

from configs.global_setting import SPAGHETTI_DIR
from src.utils.common import np2th, th2np
from src.utils.marching_cube_util import MarchingCubesMeshingTorch


@torch.no_grad()
def batch_gaus_to_gmms(gaus, device="cpu"):
    """
    Input: T(B,G,16)
    Output: [mu: T(B,1,G,3), eivec: T(B,1,G,3,3), pi: T(B,1,G), eival: T(B,1,G,3)]
    """
    gaus = np2th(gaus).to(device)
    if len(gaus.shape) < 3:
        gaus = gaus.unsqueeze(0)  # expand dim for batch

    B, G, _ = gaus.shape
    mu = gaus[:, :, :3].reshape(B, 1, G, 3)
    eivec = gaus[:, :, 3:12].reshape(B, 1, G, 3, 3)
    pi = gaus[:, :, 12].reshape(B, 1, G)
    eival = gaus[:, :, 13:16].reshape(B, 1, G, 3)

    return [mu, eivec, pi, eival]


def generate_zc_from_sj_gaus(
        spaghetti,
        sj: Union[torch.Tensor, np.ndarray],
        gaus: Union[torch.Tensor, np.ndarray],
):
    """
    Input:
        sj: [B,16,512] or [16,512]
        gaus: [B,16,16] or [16,16]
    Output:
        zc: [B,16,512]
    """
    device = spaghetti.device
    sj = np2th(sj)
    gaus = np2th(gaus)
    assert sj.dim() == gaus.dim()

    if sj.dim() == 2:
        sj = sj.unsqueeze(0)
    batch_sj = sj.to(device)
    batch_gmms = batch_gaus_to_gmms(gaus, device)
    zcs, _ = spaghetti.merge_zh(batch_sj, batch_gmms)
    return zcs


def add_spaghetti_path(spaghetti_path=SPAGHETTI_DIR):
    spaghetti_path = str(spaghetti_path)
    if spaghetti_path not in sys.path:
        sys.path.append(spaghetti_path)


def delete_spaghetti_path(
        spaghetti_path=SPAGHETTI_DIR,
):
    spaghetti_path = str(spaghetti_path)
    if spaghetti_path in sys.path:
        sys.path.remove(spaghetti_path)


def load_spaghetti(device, tag="chairs_large", model_name='spaghetti'):
    assert tag in [
        "chairs_large",
        "airplanes",
        "tables",
    ], f"tag should be 'chairs_large', 'airplanes' or 'tables'."

    add_spaghetti_path()
    from spaghetti.options import Options
    # from spaghetti.options_v1 import Options as OptionsV1
    from spaghetti.ui import occ_inference
    # from spaghetti.ui import occ_inference_v1

    # if model_name == 'spaghetti':
    opt = Options()
    # else:
    #     opt = OptionsV1()
    opt.dataset_size = 1
    opt.device = device
    opt.tag = tag
    opt.model_name = model_name
    # if model_name == 'spaghetti':
    infer_module = occ_inference.Inference(opt)
    # else:
    #     infer_module = occ_inference_v1.Inference(opt)

    spaghetti = infer_module.model.to(device)
    spaghetti.eval()
    for p in spaghetti.parameters():
        p.requires_grad_(False)
    delete_spaghetti_path()
    return spaghetti


def load_marching_cube_meshing(
        device,
        min_res=64,
):
    mesher = MarchingCubesMeshing(device=device, min_res=min_res)
    # mesher = MarchingCubesMeshingTorch(device=device, min_res=min_res)
    delete_spaghetti_path()
    return mesher


def get_mesh_from_spaghetti(spaghetti, mc_mesher, zc, res=256, model_name='spaghetti', loss_func='hinge'):
    if model_name == 'spaghetti':
        mesh = mc_mesher.occ_meshing(
            decoder=get_occ_func(spaghetti, zc), res=res, get_time=False, verbose=False
        )  # mesh (tuple) (vert, face), vert.shape = [54,3], face.shape = [104,3]
    else:
        mesh = mc_mesher.occ_meshing(
            decoder=get_occ_fun_deep_sdf(spaghetti, zc, loss_func=loss_func), res=res, get_time=False,
            verbose=False
        )  # mesh (tuple) (vert, face), vert.shape = [54,3], face.shape = [104,3]
    vert, face = list(map(lambda x: th2np(x), mesh))
    return vert, face


def get_occ_fun_deep_sdf(spaghetti, z: T, gmm: Optional[TS] = None, loss_func='hinge'):
    def forward(x: T) -> T:
        nonlocal z
        x = x.unsqueeze(0)
        out = spaghetti.occ_head(x, z, gmm)[0, :]
        if loss_func == LossType.CROSS:
            out = out.softmax(-1)
            out = -1 * out[:, 0] + out[:, 2]
        elif loss_func == LossType.IN_OUT:
            out = 2 * out.sigmoid_() - 1
        else:
            out.clamp_(-.2, .2)
        return out

    if z.dim() == 2:
        z = z.unsqueeze(0)
    return forward


def get_occ_func(spaghetti, zc):
    device = spaghetti.device
    zc = np2th(zc).to(device)

    def forward(x):
        nonlocal zc
        x = x.unsqueeze(0)
        out = spaghetti.occupancy_network(x, zc)[0, :]
        # Normalize the output to [-1, 1]
        out = 2 * out.sigmoid_() - 1
        return out

    if zc.dim() == 2:
        zc = zc.unsqueeze(0)
    return forward


def clip_eigenvalues(gaus: Union[torch.Tensor, np.ndarray], eps=1e-4):
    """
    Input:
        gaus: [B,G,16] or [G,16]
    Output:
        gaus_clipped: [B,G,16] or [G,16] torch.Tensor
    """
    gaus = np2th(gaus)
    clipped_gaus = gaus.clone()

    # Check for NaN values
    if torch.isnan(clipped_gaus).any():
        clipped_gaus = torch.where(torch.isnan(clipped_gaus), torch.tensor(eps, device=clipped_gaus.device),
                                   clipped_gaus)

    # Check for infinite values
    if torch.isinf(clipped_gaus).any():
        # Replace the infinite values with eps
        clipped_gaus = torch.where(torch.isinf(clipped_gaus), torch.tensor(eps, device=clipped_gaus.device),
                                   clipped_gaus)

    # Clamp the eigenvalues
    clipped_gaus[..., 13:16] = torch.clamp_min(clipped_gaus[..., 13:16], eps)
    return clipped_gaus


def project_eigenvectors(gaus: Union[torch.Tensor, np.ndarray]):
    """
    Input:
        gaus: [B,G,16] or [G,16]
    Output:
        gaus_projected: [B,G,16] or [1,G,16]
    """
    gaus = np2th(gaus).clone()
    if gaus.ndim == 2:
        gaus = gaus.unsqueeze(0)

    B, G = gaus.shape[:2]
    eigvec = gaus[:, :, 3:12]
    eigvec_projected = get_orthonormal_bases_svd(eigvec)
    gaus[:, :, 3:12] = eigvec_projected
    return gaus


def get_orthonormal_bases_svd(vs: torch.Tensor):
    """
    Implements the solution for the Orthogonal Procrustes problem,
    which projects a matrix to the closest rotation matrix / reflection matrix using SVD.
    Args:
        vs: Tensor of shape (B, M, 9)
    Returns:
        p: Tensor of shape (B, M, 9).
    """
    # Compute SVDs of matrices in batch
    b, m, _ = vs.shape
    vs_ = vs.reshape(b * m, 3, 3)

    U, _, Vh = torch.linalg.svd(vs_)
    # Determine the diagonal matrix to make determinants 1
    sigma = torch.eye(3)[None, ...].repeat(b * m, 1, 1).to(vs_.device)
    det = torch.linalg.det(torch.bmm(U, Vh))  # Compute determinants of UVT
    ####
    # Do not set the sign of determinants to 1.
    # Inputs contain reflection matrices.
    # sigma[:, 2, 2] = det
    ####
    # Construct orthogonal matrices
    p = torch.bmm(torch.bmm(U, sigma), Vh)
    return p.reshape(b, m, 9)


if __name__ == "__main__":
    add_spaghetti_path()
