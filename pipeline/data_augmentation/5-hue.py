#!/usr/bin/env python3
"""This module changes the hue of an image"""
import tensorflow as tf


def change_hue(image, delta):
    """This function changes the hue of an image
    Args:
        image is a 3d tf.Tensor containing the image to change
        delta is a float containing the amount the hue should change
    Returns: The adjusted image
    """
    # Change the hue of the image
    # the method adjust_hue() adjusts the hue of the image
    return tf.image.adjust_hue(image, delta)
