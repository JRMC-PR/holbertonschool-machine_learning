#!/usr/bin/env python3
"""This modlue contains the function flip_image(image)"""
import tensorflow as tf


def flip_image(image):
    """Thisa function flips an image horizontally
    Args: image is a 3d tf.tensor containing the image to flip
    Returns: The flipped image
    """
    return tf.image.flip_left_right(image)
