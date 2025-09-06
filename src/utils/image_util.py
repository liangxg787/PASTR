# -*- coding: UTF-8 -*-
"""
@Time : 16/06/2025 11:33
@Author : Xiaoguang Liang
@File : image_util.py
@Project : Enhancing_Sketch-to-3D_Controllability
"""
import os
from typing import List, Optional

import textwrap
import cv2
from PIL import Image, ImageChops, ImageDraw, ImageFont
import lxml.etree as ET
import cairosvg
import numpy as np
import multiprocessing as mp
import torchvision.transforms as T
from PIL import Image
from rembg import remove, new_session

from configs.global_setting import BASE_DIR


def stack_images_horizontally(images: List, save_path=None):
    widths, heights = list(zip(*(i.size for i in images)))
    total_width = sum(widths)
    max_height = max(heights)
    new_im = Image.new("RGBA", (total_width, max_height))

    x_offset = 0
    for im in images:
        new_im.paste(im, (x_offset, 0))
        x_offset += im.size[0]
    if save_path is not None:
        new_im.save(save_path)
    return new_im


def stack_images_vertically(images: List, save_path=None):
    widths, heights = list(zip(*(i.size for i in images)))
    max_width = max(widths)
    total_height = sum(heights)
    new_im = Image.new("RGBA", (max_width, total_height))

    y_offset = 0
    for im in images:
        new_im.paste(im, (0, y_offset))
        y_offset += im.size[1]
    if save_path is not None:
        new_im.save(save_path)
    return new_im


def merge_images(images: List, save_path=None):
    if isinstance(images[0], Image.Image):
        return stack_images_horizontally(images, save_path)

    images = list(map(stack_images_horizontally, images))
    return stack_images_vertically(images, save_path)


def draw_text(
        image: Image,
        text: str,
        font_size=None,
        font_color=(0, 0, 0),
        max_seq_length=100,
):
    W, H = image.size
    S = max(W, H)

    font_path = os.path.join(cv2.__path__[0], "qt", "fonts", "DejaVuSans.ttf")
    font_size = max(int(S / 32), 20) if font_size is None else font_size
    font = ImageFont.truetype(font_path, size=font_size)

    text_wrapped = textwrap.fill(text, max_seq_length)
    w, h = font.getsize(text_wrapped)
    new_im = Image.new("RGBA", (W, H + h))
    new_im.paste(image, (0, h))
    draw = ImageDraw.Draw(new_im)
    draw.text((max((W - w) / 2, 0), 0), text_wrapped, font=font, fill=font_color)
    return new_im


SVG_tag_prefix = "{http://www.w3.org/2000/svg}"


class SVG:

    def __init__(self, filename: str) -> None:
        self.filename = filename
        self.xml_tree = ET.parse(self.filename)
        self.root = self.xml_tree.getroot()

    def render(self, width: int, height: int, output_filename: Optional[str] = None,
               background_color="white"):
        bstr_xml = ET.tostring(self.root)
        png_data = cairosvg.svg2png(bytestring=bstr_xml, background_color=background_color,
                                    output_width=width, output_height=height)
        if output_filename is not None:
            with open(output_filename, "wb") as binary_file:
                binary_file.write(png_data)
        return png_data

    def change_width(self, new_weights: List[float]):
        if (len(new_weights) != self.num_strokes()):
            print(
                f"The list of weights has {len(new_weights)} elements and does not have the same size as the number of strokes ({self.num_strokes()} elements).")
        i = 0
        for child in self.root:
            if (child.tag == f"{SVG_tag_prefix}g"):
                for grandchild in child:
                    if (grandchild.tag == f"{SVG_tag_prefix}path"):
                        attributes = grandchild.attrib
                        attributes["stroke-width"] = f"{new_weights[i % len(new_weights)]}"

    def change_width_uniform(self, new_weight: float):
        for child in self.root:
            if (child.tag == f"{SVG_tag_prefix}g"):
                for grandchild in child:
                    if (grandchild.tag == f"{SVG_tag_prefix}path"):
                        attributes = grandchild.attrib
                        attributes["stroke-width"] = f"{new_weight}"

    def num_strokes(self):
        count = 0
        for child in self.root:
            if (child.tag == f"{SVG_tag_prefix}g"):
                for grandchild in child:
                    if (grandchild.tag == f"{SVG_tag_prefix}path"):
                        count += 1
        return count

    def tostring(self):
        return ET.tostring(self.root)


stroke_width_var = (1.0, 4.5)
cur_rand = mp.Value('i', -1)
size_random = 10000
res = 256


def get_random():
    with cur_rand.get_lock():  # Acquire the lock before modifying
        cur_rand.value += 1
        if cur_rand.value >= size_random:
            random_array = np.random.rand(size_random)
            cur_rand.value = 0
        return random_array[cur_rand.value]


def augment_svg(svg: SVG):
    width = (stroke_width_var[0] + (
            stroke_width_var[1] - stroke_width_var[0]) * get_random())
    svg.change_width_uniform(width)
    return


def svg_to_img(svg: SVG) -> Image.Image:
    # return img
    image_bytes = svg.render(res, res)
    decoded = cv2.imdecode(np.frombuffer(image_bytes, np.uint8), cv2.IMREAD_COLOR)
    image = Image.fromarray(decoded)
    return image.convert("RGB")


def build_transforms(config):
    # build train transformations
    transform_train = [
        T.Resize((config.image_size, config.image_size)),
        T.ToTensor(),
        # T.Normalize([0.5], [0.5]),
    ]

    transform_test = [
        T.Resize((config.image_size, config.image_size)),
        T.ToTensor(),
    ]

    # if config.RHFlip:
    #     transform_train += [T.RandomHorizontalFlip()]
    # if config.gblur:
    #     transform_train += [T.GaussianBlur(kernel_size=3, sigma=(0.1, 2.0))]
    # if config.center_crop_arr:
    #     transform_train += [T.CenterCrop((config.image_size, config.image_size))]

    transform_train = T.Compose(transform_train)
    transform_test = T.Compose(transform_test)

    return transform_train, transform_test


class BackgroundRemover():
    def __init__(self):
        self.session = new_session()

    def __call__(self, image: Image.Image):
        output = remove(image, session=self.session, bgcolor=[255, 255, 255, 0])
        return output


if __name__ == "__main__":
    class Config:
        image_size = 256
        RHFlip = False
        gblur = False
        center_crop_arr = None


    config = Config()

    transform_train, transform_test = build_transforms(config)
    svg_file = os.path.join(BASE_DIR,
                            "output/output_sketches/1a6f615e8b1b5ae4dbbc9440457e303e_view_-60.0/1a6f615e8b1b5ae4dbbc9440457e303e_view_-60.0_16strokes_seed0_best.svg")
    svg = SVG(svg_file)
    # augment_svg(svg)
    img = svg_to_img(svg)
    img = transform_train(img)
    img.show()
