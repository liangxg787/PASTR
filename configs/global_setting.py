# -*- coding: UTF-8 -*-
"""
@Time : 02/04/2025 09:58
@Author : Xiaoguang Liang
@File : global_setting.py
@Project : Enhancing_Sketch-to-3D_Controllability
"""
import os
from pathlib import Path

import torch
import sentry_sdk
from dotenv import load_dotenv

load_dotenv()

# ********************************************* PATH SETTING ********************************************* #

# Basic path
BASE_DIR = Path(os.path.dirname(os.path.dirname(__file__)))
SPAGHETTI_DIR = BASE_DIR / "external/spaghetti/spaghetti"
DATA_DIR = BASE_DIR / "dataset"

# ********************************************* Env setting ********************************************* #
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


class Config(object):
    # *************************************** log path setting *************************************** #

    # log path
    LOG_DIR = os.getenv('LOG_DIR', BASE_DIR / 'logs/')
    debug_log_file_path = os.path.join(LOG_DIR, "debug.log")
    info_log_file_path = os.path.join(LOG_DIR, "info.log")
    error_log_file_path = os.path.join(LOG_DIR, "error.log")

    # log form
    log_format = "{time:YYYY-MM-DD HH:mm:sss} | {message}"

    # *************************************** sentry setting *************************************** #

    SENTRY_URL = os.getenv('SENTRY_URL', '')


class Development(Config):
    pass


class Production(Config):
    pass


# Set the default settings
settings = {
    "default": Config,
    "development": Development,
    "production": Production
}

# Get the environment configuration key from the environment variable
SETTINGS = settings[os.getenv('ENV', 'default')]

# Capture the error with Sentry
sentry_sdk.init(SETTINGS.SENTRY_URL)

if __name__ == '__main__':
    print(settings['default'].debug_log_file_path)
    print(settings['development'].debug_log_file_path)
    print(settings['production'].debug_log_file_path)
