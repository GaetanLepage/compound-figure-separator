"""
#############################
#        CompFigSep         #
# Compound Figure Separator #
#############################

GitHub:         https://github.com/GaetanLepage/compound-figure-separator

Author:         Gaétan Lepage
Email:          gaetan.lepage@grenoble-inp.org
Date:           Spring 2020

Master's project @HES-SO (Sierre, SW)

Supervisors:    Henning Müller (henning.mueller@hevs.ch)
                Manfredo Atzori (manfredo.atzori@hevs.ch)

Collaborator:   Niccolò Marini (niccolo.marini@hevs.ch)


########################################################
Miscellaneous functions related to TensorFlow data sets.

Those functions are deprecated since the comp-fig-sep project is now using the Detectron2 API
which is based on PyTorch.
If you would like to do experiments with TensorFlow, please, feel free to use those.
"""

from typing import List

import tensorflow as tf


def int64_feature(value: int) -> tf.train.Feature:
    """
    Wrapper to create a TensorFlow int64 feature.

    Args:
        value (int):    The value to convert.

    Returns:
        feature (tf.train.Feature): TensorFlow int64 feature.
    """
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))


def int64_list_feature(value: List[int]) -> tf.train.Feature:
    """
    Wrapper to create a TensorFlow int64 list feature.

    Args:
        value (List[int]):  The value to convert.

    Returns:
        feature (tf.train.Feature): TensorFlow int64 list feature.
    """
    return tf.train.Feature(int64_list=tf.train.Int64List(value=value))


def bytes_feature(value: str) -> tf.train.Feature:
    """
    Wrapper to create a TensorFlow byte feature.

    Args:
        value (byte string):    The value to convert.

    Returns:
        feature (tf.train.Feature): TensorFlow byte feature.
    """
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))


def bytes_list_feature(value: List[str]) -> tf.train.Feature:
    """
    Wrapper to create a TensorFlow byte list feature.

    Args:
        value (List[byte string]):  A list of bytes.

    Returns:
        feature (tf.train.Feature): TensorFlow byte list feature.
    """
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=value))


def float_list_feature(value: List[float]) -> tf.train.Feature:
    """
    Wrapper to create a TensorFlow float list feature.

    Args:
        value (List[float]):    A list of float.

    Returns:
        feature (tf.train.Feature): TensorFlow float list feature.
    """
    return tf.train.Feature(float_list=tf.train.FloatList(value=value))
