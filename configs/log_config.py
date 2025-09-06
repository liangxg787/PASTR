# -*- coding: UTF-8 -*-
"""
@Time : 02/04/2025 09:59
@Author : Xiaoguang Liang
@File : log_config.py
@Project : Enhancing_Sketch-to-3D_Controllability
"""
import logging

# Create a logger
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

# Create a file handler
# file_handler = logging.FileHandler('')
# file_handler.setLevel(logging.DEBUG)

# Create a console handler
console_handler = logging.StreamHandler()
console_handler.setLevel(logging.DEBUG)

# Define the formatter
# formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')

# file_handler.setFormatter(formatter)
console_handler.setFormatter(formatter)

# Add handlers
# logger.addHandler(handler)
logger.addHandler(console_handler)

if __name__ == '__main__':
    logger.debug('This is a debug message')
    logger.info('This is an info message')
