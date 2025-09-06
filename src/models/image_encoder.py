# -*- coding: UTF-8 -*-
"""
@Time : 21/06/2025 11:07
@Author : Xiaoguang Liang
@File : image_encoder.py
@Project : Enhancing_Sketch-to-3D_Controllability
"""
from transformers import AutoImageProcessor, ResNetModel, ViTImageProcessor, ViTModel, Dinov2Model, AutoModel

from configs.global_setting import device


class ResNetEncoder(object):

    def __init__(self):
        super(ResNetEncoder).__init__()
        # model_name = "microsoft/resnet-50"
        # model_name = "microsoft/resnet-101"
        model_name = "microsoft/resnet-152"

        self.image_processor = AutoImageProcessor.from_pretrained(model_name)
        self.model = ResNetModel.from_pretrained(model_name)
        self.model.to(device)

    def encode(self, image):
        inputs = self.image_processor(image, return_tensors="pt")
        inputs = inputs.to(device)
        outputs = self.model(**inputs)
        return outputs


class ViTEncoder(ResNetEncoder):

    def __init__(self):
        super(ViTEncoder).__init__()
        model_name = "google/vit-base-patch16-224-in21k"
        self.image_processor = ViTImageProcessor.from_pretrained(model_name)
        self.model = ViTModel.from_pretrained(model_name)
        self.model.to(device)


class DinoEncoderV2Base(ResNetEncoder):

    def __init__(self):
        super(DinoEncoderV2Base).__init__()
        model_name = "facebook/dinov2-base"
        self.image_processor = AutoImageProcessor.from_pretrained(model_name)
        self.model = Dinov2Model.from_pretrained(model_name)
        self.model.to(device)


class DinoEncoderV2Large(ResNetEncoder):

    def __init__(self):
        super(DinoEncoderV2Large).__init__()
        model_name = "facebook/dinov2-large"
        self.image_processor = AutoImageProcessor.from_pretrained(model_name)
        self.model = Dinov2Model.from_pretrained(model_name)
        self.model.to(device)


class DinoEncoderV3Vitb16(ResNetEncoder):

    def __init__(self):
        super(DinoEncoderV3Vitb16).__init__()
        model_name = "facebook/dinov3-vitb16-pretrain-lvd1689m"
        self.image_processor = AutoImageProcessor.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name)
        self.model.to(device)


class DinoEncoderV3Vitl16(ResNetEncoder):

    def __init__(self):
        super(DinoEncoderV3Vitl16).__init__()
        model_name = "facebook/dinov3-vitl16-pretrain-lvd1689m"
        self.image_processor = AutoImageProcessor.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name)
        self.model.to(device)
