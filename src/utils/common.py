# -*- coding: UTF-8 -*-
"""
@Time : 16/06/2025 11:28
@Author : Xiaoguang Liang
@File : common.py
@Project : Enhancing_Sketch-to-3D_Controllability
"""
import time
from contextlib import contextmanager
from functools import partial

import numpy as np
import torch

from configs.log_config import logger


def th2np(tensor):
    if isinstance(tensor, np.ndarray):
        return tensor
    if isinstance(tensor, torch.Tensor):
        return tensor.detach().cpu().numpy()


def np2th(ndarray):
    if isinstance(ndarray, torch.Tensor):
        return ndarray.detach().cpu()
    elif isinstance(ndarray, np.ndarray):
        return torch.tensor(ndarray).float()
    else:
        raise ValueError("Input should be either torch.Tensor or np.ndarray")


@contextmanager
def timer(msg="all tasks"):
    """
    Calculate the time of running
    @return:
    """
    startTime = time.time()
    yield
    endTime = time.time()
    time_span = endTime - startTime
    if time_span < 1:
        cost_time = round(time_span * 1000, 2)
        logger.info(f"The time cost for {msg}：{cost_time} ms")
    elif time_span < 60:
        cost_time = round(time_span, 2)
        logger.info(f"The time cost for {msg}：{cost_time} s")
    else:
        cost_time = round(time_span / 60, 2)
        logger.info(f"The time cost for {msg}：{cost_time} minutes")


def pickle_able_wrapper(cls):
    """
    Wrap a class that does not support pickle serialization into a new "class" that supports pickle serialization

    Parameters
    ----------
    cls : class
        The original class object that needs to be wrapped

    Returns
    -------
    wrapper : class
        The decorated new class is identical to the original class except that it supports pickle serialization and deserialization.
    """
    return partial(PickleAbleWrapper, cls)


class PickleAbleWrapper:
    """
        Wrap a class that does not support pickle serialization into a new "class" that supports pickle serialization
    Parameters
    ----------
    aim_cls : class
        he original class object that needs to be wrapped
    *args
        Position parameters
    **kwargs
        keyword arguments

    Notes
    -----
    1. This class can encapsulate classes that are inconvenient to modify to support pickle serialization.
    2. The encapsulated class (i.e., PickleWrapper itself) automatically inherits all attributes and methods
    except the __init__() method from the original class, while also supporting pickle serialization.
    This allows instances of the encapsulated class to be saved to files or sent to another process.
    """

    def __init__(self, cls, *args, **kwargs):
        self.__dict__["_cls"] = cls
        self.__dict__["_args"] = args
        self.__dict__["_kwargs"] = kwargs
        self.__dict__["_wrapped"] = cls(*args, **kwargs)

    def __getattr__(self, item):
        """Get the properties or methods of the current class"""
        return getattr(self.__dict__["_wrapped"], item)

    def __getstate__(self):
        """During serialization, the instance of the encapsulated class is not preserved; instead,
        the stored encapsulated class and its constructor parameters are serialized."""
        return self.__dict__["_cls"], self.__dict__["_args"], self.__dict__["_kwargs"]

    def __setstate__(self, state):
        """Deserialize the encapsulated class and its constructor parameters, and store them in the encapsulated class."""
        cls, args, kwargs = state
        self.__dict__["_cls"] = cls
        self.__dict__["_args"] = args
        self.__dict__["_kwargs"] = kwargs
        self.__dict__["_wrapped"] = cls(*args, **kwargs)
