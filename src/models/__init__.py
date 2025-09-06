# -*- coding: UTF-8 -*-
"""
@Time : 16/06/2025 18:00
@Author : Xiaoguang Liang
@File : __init__.py.py
@Project : Enhancing_Sketch-to-3D_Controllability
"""
from .image_encoder import ResNetEncoder, ViTEncoder, DinoEncoderV2Base, DinoEncoderV2Large, \
    DinoEncoderV3Vitb16, DinoEncoderV3Vitl16

__model_factory = {
    "encoder_resnet": ResNetEncoder,
    "encoder_vit": ViTEncoder,
    "encoder_dino_v2_base": DinoEncoderV2Base,
    "encoder_dino_v2_large": DinoEncoderV2Large,
    "encoder_dino_v3_vitb16": DinoEncoderV3Vitb16,
    "encoder_dino_v3_vitl16": DinoEncoderV3Vitl16,
}


def get_names():
    return list(__model_factory.keys())


def init_model(name, *args, **kwargs):
    if name not in list(__model_factory.keys()):
        raise KeyError(f"Model type '{name}' is not supported")
    return __model_factory[name](*args, **kwargs)
