# -*- coding: UTF-8 -*-
"""
@Time : 02/04/2025 08:48
@Author : Xiaoguang Liang
@File : 2D_projection.py
@Project : Enhancing_Sketch-to-3D_Controllability
"""
import os
import math
import shutil
from fnmatch import fnmatch
from functools import partial

import torch
from pytorch3d.io import load_obj, load_objs_as_meshes
from pytorch3d.structures import Meshes
from pytorch3d.renderer import (
    look_at_view_transform,
    FoVPerspectiveCameras,
    PointLights,
    RasterizationSettings,
    MeshRenderer,
    MeshRasterizer,
    SoftPhongShader,
    HardPhongShader,
    SoftSilhouetteShader,
    TexturesVertex,
)
import matplotlib.pyplot as plt
from tqdm import tqdm
from joblib import Parallel, delayed

from configs.global_setting import BASE_DIR
from src.utils.common import timer
from configs.log_config import logger
from configs.global_setting import device


def project_2d(obj_filename, save_dir=None):
    # Load model
    verts, faces_idx, _ = load_obj(obj_filename)
    faces = faces_idx.verts_idx
    # Add texture to the mesh
    # Use the white texture
    textures = TexturesVertex(verts_features=torch.ones_like(verts)[None])
    mesh = Meshes(verts=[verts], faces=[faces], textures=textures)

    mesh = mesh.to(device)

    # Define a list of (elevation, azimuth) angles for different views
    # viewpoints = [
    #     (0, 0),  # Front view
    #     (0, 90),  # Side view
    #     (0, 180),  # Back view
    #     (0, 270),  # Other side view
    #     (90, 0),  # Top view
    #     (-90, 0),  # Bottom view
    #     (45, 45),  # Diagonal view
    # ]

    # render from 6 distinct views with azimuthal angles distributed across
    # -π,π) and at a constant elevation of Pi/10 from a distance of 2.07 units.
    # Set the elevation to Pi/10 and distance to 2.07
    # elevation = math.pi / 10
    elevation = 30
    # distance = 2.07
    distance = 1

    # Set the light position
    lights = PointLights(device=device, location=((0, 1, 0),))

    # Set the rasterization settings
    raster_settings = RasterizationSettings(
        image_size=512,
        blur_radius=0.0,
        faces_per_pixel=1,
        bin_size=0,
    )

    # Define a list of azimuthal angles for different views
    # 6 views evenly distributed between -π and π
    azimuths = [i * 2 * 180 / 6 - 180 for i in range(6)]

    # for i, (elev, azim) in enumerate(viewpoints):
    for i, azim in enumerate(azimuths):
        R, T = look_at_view_transform(dist=distance, elev=elevation, azim=azim)
        cameras = FoVPerspectiveCameras(device=device, R=R, T=T)

        # Set the shader
        shader = SoftPhongShader(device=device, cameras=cameras, lights=lights)

        # Create a renderer
        renderer = MeshRenderer(
            rasterizer=MeshRasterizer(
                cameras=cameras,
                raster_settings=raster_settings
            ),
            shader=shader
        )

        images = renderer(mesh, cameras=cameras)
        # Get the RGB image
        image = images[0, ..., :3].cpu().numpy()
        # save_file = os.path.join(save_dir, f'view_{i}.png')
        shape_name = obj_filename.split('/')[-2]
        save_file = f'{save_dir}/{shape_name}_view_{azim}.png'
        plt.imsave(save_file, image)


def parallel_project_2d(obj_dir, save_dir, pattern="*manifold.obj", cpu_num=10):
    args = []
    for path, subdirs, files in os.walk(obj_dir):
        for name in files:
            if fnmatch(name, pattern):
                file_path = os.path.join(path, name)
                args.append(file_path)

    logger.info(f"The amount of files: {len(args)}")

    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    project_2d_ = partial(project_2d, save_dir=save_dir)

    # for batch in tqdm(args):
    #     project_2d_(batch)

    Parallel(n_jobs=cpu_num, backend='loky', verbose=10)(
        delayed(project_2d_)(batch) for batch in tqdm(args))


def split_projection_dataset(folder, split_amount=800):
    i = 0
    j = 1
    for path, subdirs, files in os.walk(folder):
        for name in files:
            if fnmatch(name, "*.png"):
                i += 1
                new_folder = f"{folder}/{j}"
                if not os.path.exists(new_folder):
                    os.makedirs(new_folder)
                if i <= split_amount * j:
                    # Move the file in the new directory
                    old_path = str(os.path.join(path, name))
                    shutil.move(old_path, new_folder)
                if i == split_amount * j:
                    j += 1


if __name__ == "__main__":
    # Path to the OBJ file
    filename = BASE_DIR / "dataset/03001627_10/1a6f615e8b1b5ae4dbbc9440457e303e/model_manifold.obj"
    # filename = BASE_DIR / "dataset/03001627_10/1a6f615e8b1b5ae4dbbc9440457e303e/model.obj"
    # save_path = BASE_DIR / "output/2d_projection/"
    save_path = BASE_DIR / "output/03001627_2d_projection/"
    # project_2d(str(filename), save_path)

    # shapenet_folder = BASE_DIR / "dataset/03001627_10/"
    shapenet_folder = BASE_DIR / "dataset/03001627/"
    # with timer("All tasks"):
    #     parallel_project_2d(shapenet_folder, save_path, cpu_num=16)

    split_projection_dataset(save_path)
