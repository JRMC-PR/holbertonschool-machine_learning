#!/usr/bin/env python3
"""This module rotes an image 90 degrees counter-clockwise"""
import tensorflow as tf


def rotate_image(image):
    """This function rotates an image 90 degrees counter-clockwise
    Args: image is a 3d tf.Tensor containing the image to rotate
    Returns: The rotated image
    """
    # Rotate the image counter-clockwise
    # the method rotate() rotates the image counter-clockwise
    return tf.image.rot90(image)
