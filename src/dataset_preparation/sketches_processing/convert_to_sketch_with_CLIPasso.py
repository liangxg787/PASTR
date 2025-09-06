# -*- coding: UTF-8 -*-
"""
@Time : 19/06/2025 16:54
@Author : Xiaoguang Liang
@File : convert_to_sketch_with_CLIPasso.py
@Project : Enhancing_Sketch-to-3D_Controllability
"""
'''
Run this script in CLIPasso
Reference: https://github.com/yael-vinker/CLIPasso.git
'''
import os
import shutil
from fnmatch import fnmatch
import traceback

from tqdm import tqdm
from joblib import Parallel, delayed
from sentry_sdk import capture_exception

from src.utils.common import timer
from configs.log_config import logger
from configs.global_setting import BASE_DIR

# STROKES = 8
# STROKES = 16
STROKES = 32


def to_sketch(target_file):
    try:
        os.system(
            f"python run_object_sketching.py --target_file {target_file} --num_strokes {STROKES} --num_iter 2000 --fix_scale 1 --num_sketches 1")
        logger.info(f"Done convert sketches for {target_file}!")
    except:
        traceback.print_exc()


def parallel_to_sketch(data_path, save_path, pattern="*.png", cpu_num=3):
    if not os.path.exists(data_path):
        raise FileNotFoundError(f"{data_path} does not exist!")
    processed_images = []
    if os.path.exists(save_path):
        for sub_folder in os.listdir(save_path):
            processed_images.append(sub_folder)
    else:
        logger.info(f"{save_path} does not exist!")
    logger.info(f"The amount of processed images: {len(processed_images)}")

    args = []
    for path, subdirs, files in os.walk(data_path):
        for name in files:
            if fnmatch(name,
                       pattern):
                file_path = os.path.join(name)
                image_name = name.replace(".png", "")
                if image_name not in processed_images:
                    args.append(file_path)

    logger.info(f"The amount of files: {len(args)}")

    # for batch in tqdm(args):
    #     to_sketch(batch)

    Parallel(n_jobs=cpu_num, backend='multiprocessing', verbose=10)(
        delayed(to_sketch)(batch) for batch in tqdm(args))


def filter_sketches(save_path, target_folder, pattern="*.png"):
    all_image_names = []
    for path, subdirs, files in os.walk(target_folder):
        for name in files:
            if fnmatch(name,
                       pattern):
                image_name = name.replace(".png", "")
                all_image_names.append(image_name)

    new_folder = save_path.replace("output_sketches", "output_sketches_filtered")
    if not os.path.exists(new_folder):
        os.makedirs(new_folder)

    for sub_folder in os.listdir(save_path):
        sub_folder_path = os.path.join(save_path, sub_folder)
        if sub_folder not in all_image_names:
            shutil.move(sub_folder_path, new_folder)


def move_svg(target_folder, save_path, pattern="*.png"):
    all_image_names = []
    for path, subdirs, files in os.walk(target_folder):
        for name in files:
            if fnmatch(name,
                       pattern):
                image_name = name.replace(".png", "")
                all_image_names.append(image_name)

    new_folder = save_path.replace("output_sketches", "sketch_svg")
    if not os.path.exists(new_folder):
        os.makedirs(new_folder)

    log_file = os.path.join(new_folder, "failed_svg_log.txt")
    log_f = open(log_file, "w")

    for sub_folder in os.listdir(save_path):
        sub_folder_path = os.path.join(save_path, sub_folder)
        if sub_folder in all_image_names:
            for file in os.listdir(sub_folder_path):
                if file.endswith("best.svg"):
                    shutil.copy(os.path.join(sub_folder_path, file), os.path.join(new_folder, file))
                    break
            else:
                logger.info(f"{sub_folder} does not have best.svg!")
                log_f.write(f"{sub_folder}\n")
                image_name = sub_folder + '.png'
                to_sketch(image_name)
            all_image_names.remove(sub_folder)

    if all_image_names:
        # for sub_folder in all_image_names:
        #     logger.info(f"{sub_folder} does not exist in output_sketches!")
        #     log_f.write(f"{sub_folder}\n")
        #     image_name = sub_folder + '.png'
        #     to_sketch(image_name)
        cpu_num = 3
        logger.info(f"{len(all_image_names)} image names do not exist in output_sketches!")
        all_image_names_path = [x + '.png' for x in all_image_names]
        Parallel(n_jobs=cpu_num, backend='multiprocessing', verbose=10)(
            delayed(to_sketch)(batch) for batch in tqdm(all_image_names_path))


def merge_svg_files(svg_path, new_folder):
    if not os.path.exists(new_folder):
        os.mkdir(new_folder)
    for path, subdirs, files in os.walk(svg_path):
        for name in files:
            if fnmatch(name,
                       "*.svg"):
                svg_file = os.path.join(path, name)
                shutil.copy(svg_file, os.path.join(new_folder, name))


if __name__ == '__main__':
    root = '/scratch/xl01339/Enhancing_Sketch-to-3D_Controllability/output/2d_projection'
    # pattern = "*.obj"
    save_path = '/scratch/xl01339/CLIPasso/output_sketches'
    target_folder = '/scratch/xl01339/CLIPasso/target_images'

    # sketch_path = BASE_DIR / 'dataset/clipasso_sketches_8strokes'
    # new_sketch_path = BASE_DIR / 'dataset/clipasso_8strokes'
    # sketch_path = BASE_DIR / 'dataset/clipasso_sketches_16strokes'
    # new_sketch_path = BASE_DIR / 'dataset/clipasso_16strokes'
    sketch_path = BASE_DIR / 'dataset/clipasso_sketches_32strokes'
    new_sketch_path = BASE_DIR / 'dataset/clipasso_32strokes'

    try:
        with timer("all tasks"):
            # parallel_to_sketch(root, save_path)

            # filter_sketches(save_path, target_folder)
            # move_svg(target_folder, save_path)

            merge_svg_files(sketch_path, new_sketch_path)

    except Exception as exc:
        capture_exception(exc)
        logger.error(exc)
        raise exc
