# -*- coding: UTF-8 -*-
"""
@Time : 07/07/2025 22:41
@Author : Xiaoguang Liang
@File : evaluate_model.py
@Project : Enhancing_Sketch-to-3D_Controllability
"""
import os
import gc
from datetime import datetime
from functools import partial

import torch
from tqdm import tqdm, trange
from pytorch3d.io import save_obj, load_obj
from pytorch3d.structures import Meshes
from pytorch3d.ops import sample_points_from_meshes
from joblib import Parallel, delayed

from src.metrics.evaluation_metrics_v1 import compute_all_metrics
# from src.metrics.evaluation_metrics_v2 import compute_all_metrics
from src.utils.data_util import get_sketch_data, get_test_dataset, down_sample, get_amateur_sketch_data
from src.utils.spaghetti_util import (generate_zc_from_sj_gaus,
                                      get_mesh_from_spaghetti, load_marching_cube_meshing,
                                      load_spaghetti)
from configs.global_setting import DATA_DIR, device
from configs.log_config import logger

RESOLUTION = 256
cpu_num = 2


def evaluate_model(model, test_dataset_dir, sketch_path, num_samples=3, num_sample_points=2048,
                   pattern="*.png", time_stamp='', batch_size=10, chunk_size=500):
    if not time_stamp:
        # Get the time stamp now
        time_stamp = datetime.now().strftime("%m%d_%H%M%S")
        final_obj_dir = DATA_DIR / f'predicted_meshes/{time_stamp}'

        model.to(device)
        model.stage1_model.to(device)
        test_objs = get_test_dataset(test_dataset_dir)

        if 'Amateur' in str(sketch_path):
            sketch_data = get_amateur_sketch_data(sketch_path, pattern=pattern)
        else:
            sketch_data = get_sketch_data(sketch_path, pattern=pattern)

        if num_samples:
            sample_meshes = down_sample(test_objs, num_samples=num_samples)
        else:
            sample_meshes = list(test_objs.keys())

        test_images = list(map(lambda x: sketch_data[x][0], sample_meshes))

        for chunk in trange(0, len(test_images), chunk_size):
            chunk_images = test_images[chunk:chunk + chunk_size]
            # Stage1 sampling
            classifier_free_guidance = False

            free_guidance_weight = 1.0
            num_inference_steps = 50
            extrinsics = model.stage1_model.sampling_gaussians(chunk_images,
                                                               classifier_free_guidance=classifier_free_guidance,
                                                               free_guidance_weight=free_guidance_weight,
                                                               num_inference_steps=num_inference_steps)
            # stage2 sampling
            intrinsics = model.sample(chunk_images, extrinsics,
                                      classifier_free_guidance=classifier_free_guidance,
                                      free_guidance_weight=free_guidance_weight,
                                      num_inference_steps=num_inference_steps
                                      )

            # Load spaghetti
            if model.hparams.spaghetti_tag == 'chairs_large_6755':
                spaghetti_tag = 'chairs_large'
            else:
                spaghetti_tag = model.hparams.spaghetti_tag
            spaghetti = load_spaghetti(device=device, tag=spaghetti_tag)
            zcs = generate_zc_from_sj_gaus(spaghetti, intrinsics, extrinsics)

            logger.info("Start to generate and save meshes ...")
            # _generate_and_save_mesh = partial(generate_and_save_mesh, model=model, sample_meshes=sample_meshes,
            #                                   final_obj_dir=final_obj_dir)
            # Parallel(n_jobs=cpu_num, backend='loky', verbose=10)(
            #     delayed(_generate_and_save_mesh)(batch) for batch in tqdm(enumerate(zcs)))
            for i, x in tqdm(enumerate(zcs)):
                try:
                    generated_mesh = get_mesh_from_spaghetti(spaghetti, model.mc_mesher, x, res=RESOLUTION)
                except Exception as e:
                    logger.error(f"Mesh generation failed: {e}")
                    continue

                # Fetch the verts and faces of the final predicted mesh
                final_verts, final_faces = generated_mesh
                final_verts = torch.from_numpy(final_verts)
                final_faces = torch.from_numpy(final_faces)

                # Store the predicted mesh using save_obj
                j = i + chunk
                mesh_name = sample_meshes[j]
                # temp_final_obj_dir = final_obj_dir / f'{mesh_name}'
                temp_final_obj_dir = final_obj_dir

                if not os.path.exists(temp_final_obj_dir):
                    os.makedirs(temp_final_obj_dir)
                final_obj = temp_final_obj_dir / f'{mesh_name}.obj'
                save_obj(final_obj, final_verts, final_faces)

        torch.cuda.empty_cache()
        del model
        del spaghetti
        del sketch_data
        del sample_meshes
        del test_images
        del zcs
        del extrinsics
        del intrinsics
        gc.collect()

    # Get the results of metrics
    logger.info("Evaluating metrics...")
    final_obj_dir = DATA_DIR / f'predicted_meshes/{time_stamp}'
    test_objs = get_test_dataset(final_obj_dir, pattern="*.obj")
    sample_meshes = list(test_objs.keys())
    a_test_dataset, b_test_dataset = make_test_dataset(test_dataset_dir,
                                                       final_obj_dir,
                                                       sample_meshes=sample_meshes,
                                                       num_sample_points=num_sample_points)

    del test_objs
    del sample_meshes
    gc.collect()

    a_test_dataset = a_test_dataset.to(device)
    b_test_dataset = b_test_dataset.to(device)
    logger.info("Start computing metrics")
    results = compute_all_metrics(a_test_dataset, b_test_dataset, batch_size)
    logger.info(f'results: {results}')
    return results


def generate_and_save_mesh(idx_zc, model=None, sample_meshes=[], final_obj_dir=''):
    idx, zc = idx_zc
    try:
        spaghetti = load_spaghetti(device=device, tag=model.hparams.spaghetti_tag)
        generated_mesh = get_mesh_from_spaghetti(spaghetti, model.mc_mesher, zc, res=RESOLUTION)
    except Exception as e:
        logger.error(f"Mesh generation failed: {e}")
        generated_mesh = None

    if generated_mesh:
        # Fetch the verts and faces of the final predicted mesh
        final_verts, final_faces = generated_mesh
        final_verts = torch.from_numpy(final_verts)
        final_faces = torch.from_numpy(final_faces)

        # Store the predicted mesh using save_obj
        mesh_name = sample_meshes[idx]
        temp_final_obj_dir = final_obj_dir / f'{mesh_name}'

        if not os.path.exists(temp_final_obj_dir):
            os.makedirs(temp_final_obj_dir)
        final_obj = temp_final_obj_dir / f'{mesh_name}.obj'
        save_obj(final_obj, final_verts, final_faces)


def make_test_dataset(a_objs_path, b_objs_path, sample_meshes=[], num_sample_points=2048):
    a_objs = get_test_dataset(a_objs_path)
    b_objs = get_test_dataset(b_objs_path, pattern="*.obj")
    if not sample_meshes:
        sample_meshes = down_sample(a_objs)

    a_test_dataset = make_points_from_meshes(a_objs, sample_meshes, num_sample_points)
    b_test_dataset = make_points_from_meshes(b_objs, sample_meshes, num_sample_points)

    return a_test_dataset, b_test_dataset


def get_mesh_from_obj(trg_obj):
    # We read the target 3D model using load_obj
    verts, faces, aux = load_obj(trg_obj)

    # verts is a FloatTensor of shape (V, 3) where V is the number of vertices in the mesh
    # faces is an object which contains the following LongTensors: verts_idx, normals_idx and textures_idx
    # For this tutorial, normals and textures are ignored.
    faces_idx = faces.verts_idx
    verts = verts

    # We scale normalize and center the target mesh to fit in a sphere of radius 1 centered at (0,0,0).
    # (scale, center) will be used to bring the predicted mesh to its original center and scale
    # Note that normalizing the target mesh, speeds up the optimization but is not necessary!
    center = verts.mean(0)
    verts = verts - center
    scale = max(verts.abs().max(0)[0])
    verts = verts / scale

    return verts, faces_idx


def make_points_from_meshes(objs, sample_meshes, num_sample_points):
    all_verts = []
    all_faces = []
    for mesh_name in tqdm(sample_meshes):
        obj_path = objs[mesh_name]
        obj_path = str(obj_path)
        verts, faces_idx = get_mesh_from_obj(obj_path)
        all_verts.append(verts)
        all_faces.append(faces_idx)

    # We construct a Meshes structure for the target mesh
    trg_mesh = Meshes(verts=all_verts, faces=all_faces)
    sample_points = sample_points_from_meshes(trg_mesh, num_sample_points)
    return sample_points
