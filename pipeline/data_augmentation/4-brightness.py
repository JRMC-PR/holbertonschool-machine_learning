#!/usr/bin/env python3
"""This module contains the function change_brightness(image, max_delta)"""
import tensorflow as tf


def change_brightness(image, max_delta):
    """This function adjusts the brightness of an image
    Args:
        image is a 3d tf.Tensor containing the image to change
        max_delta is a float containing the maximum amount the image should be
        brightened (or darkened)
    Returns: The adjusted image
    """
    # Adjust the brightness of the image
    # the method adjust_brightness() adjusts the brightness of the image
    return tf.image.adjust_brightness(image, max_delta)
