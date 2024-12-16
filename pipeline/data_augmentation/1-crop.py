#!/usr/bin/env python3
"""This module randomly crops an image"""
import tensorflow as tf


def crop_image(image, size):
    """This function randomly crops an image
    Args:
        image is a 3d tf.Tensor containing the image to crop
        size is a tuple containing the crop size
    Returns: The cropped image
    """
    # Randomly crop the image
    # the method random_crop() crops the image to the specified size
    return tf.image.random_crop(image, size)
