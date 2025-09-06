# -*- coding: UTF-8 -*-
"""
@Time : 01/07/2025 10:00
@Author : Xiaoguang Liang
@File : marching_cube_util.py
@Project : Enhancing_Sketch-to-3D_Controllability
"""
import time

import torch
from pytorch3d.ops.marching_cubes import marching_cubes
from spaghetti.custom_types import *
from spaghetti.utils.mcubes_meshing import MarchingCubesMeshing

from configs.log_config import logger


def mcubes_torch(pytorch_3d_occ_tensor: T, voxel_grid_origin: List[float], voxel_size: float) -> T_Mesh:
    # verts, faces = marching_cubes(pytorch_3d_occ_tensor, 0)
    # print(pytorch_3d_occ_tensor.shape)
    verts, faces = marching_cubes(pytorch_3d_occ_tensor.unsqueeze(0), isolevel=0)
    verts = verts[0]
    faces = faces[0]
    mesh_points = torch.zeros_like(verts)
    mesh_points[:, 0] = voxel_grid_origin[0] + verts[:, 0]
    mesh_points[:, 1] = voxel_grid_origin[1] + verts[:, 1]
    mesh_points[:, 2] = voxel_grid_origin[2] + verts[:, 2]
    return mesh_points, faces


class MarchingCubesMeshingTorch(MarchingCubesMeshing):

    def occ_meshing(self, decoder, res: int = 256, get_time: bool = False, verbose=False):
        start = time.time()
        voxel_origin = [-1., -1., -1.]
        voxel_size = 2.0 / (res - 1)
        occ_values = self.get_grid(decoder, res)
        if verbose:
            end = time.time()
            logger.info("sampling took: %f" % (end - start))
            if get_time:
                return end - start

        mesh_a = mcubes_torch(occ_values, voxel_origin, voxel_size)

        if verbose:
            end_b = time.time()
            logger.info("mcube took: %f" % (end_b - end))
            logger.info("meshing took: %f" % (end_b - start))
        return mesh_a
