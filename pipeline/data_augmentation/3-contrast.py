#!/usr/bin/env python3
"""This module randomly adjusts the contrast of an image"""
import tensorflow as tf


def change_contrast(image, lower, upper):
    """This function changes the contrast of an image
    Args:
        image is a 3d tf.Tensor containing the image to change
        lower is a float containing the lower bound of the contrast
        upper is a float containing the upper bound of the contrast
    Returns: The adjusted image
    """
    return tf.image.random_contrast(image, lower, upper)
