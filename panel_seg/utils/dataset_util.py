"""
Miscellaneous functions related to TensorFlow data sets.
"""

import tensorflow as tf

from typing import List


def int64_feature(value: int):
    """
    Wrapper to create a TensorFlow int64 feature.

    Args:
        value (int): The value to convert.
    """
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))


def int64_list_feature(value: List[int]):
    """
    Wrapper to create a TensorFlow int64 list feature.

    Args:
        value (List[int]): The value to convert.
    """
    return tf.train.Feature(int64_list=tf.train.Int64List(value=value))


def bytes_feature(value: str):
    """
    Wrapper to create a TensorFlow byte feature.

    Args:
        value (byte string): The value to convert.
    """
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))


def bytes_list_feature(value: List[str]):
    """
    Wrapper to create a TensorFlow byte list feature.

    Args:
        value (List[byte string]): A list of bytes.
    """
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=value))


def float_list_feature(value: List[float]):
    """
    Wrapper to create a TensorFlow float list feature.

    Args:
        value (List[float]): A list of bytes.
    """
    return tf.train.Feature(float_list=tf.train.FloatList(value=value))
