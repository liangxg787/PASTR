# -*- coding: UTF-8 -*-
"""
@Time : 24/06/2025 20:54
@Author : Xiaoguang Liang
@File : process_ids.py
@Project : Enhancing_Sketch-to-3D_Controllability
"""
import json
from collections import defaultdict

import pandas as pd

from configs.global_setting import BASE_DIR


def process_ids(file_path, save_path):
    data = pd.read_csv(file_path)
    data = data[['sn', 'spaghetti_idx']]
    ids_dict = defaultdict(tuple)
    for i, row in data.iterrows():
        ids_dict[i] = (row['sn'], row['spaghetti_idx'])
    with open(save_path, 'w') as f:
        json.dump(ids_dict, f)


if __name__ == '__main__':
    file_p = BASE_DIR / 'dataset/salad_data/autosdf_spaghetti_intersec_game_data.csv'
    save_p = BASE_DIR / 'dataset/salad_data/ids.json'
    process_ids(file_p, save_p)
