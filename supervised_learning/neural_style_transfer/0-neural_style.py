#!/usr/bin/env python3
"""This module contain the clas NST
"""
import numpy as np
import tensorflow as tf


class NST:
    """This is the class NST"""

    # Public class attributes
    style_layers = [
        "block1_conv1",
        "block2_conv1",
        "block3_conv1",
        "block4_conv1",
        "block5_conv1",
    ]
    content_layer = "block5_conv2"

    # Class constructor
    def __init__(self, style_image, content_image, alpha=1e4, beta=1):
        """Initializer
        Arguments:
            style_image {np.ndarray} -- the image style
            content_image {np.ndarray} -- the image content
            alpha {float} -- the weight for style cost
            beta {float} -- the weight for content cost
        """
        if (
            type(style_image) is not np.ndarray
            or style_image.ndim != 3
            or type(content_image) is not np.ndarray
            or content_image.ndim != 3
        ):
            raise TypeError(
                "style_image and content_image \
                    must be np.ndarray with shape (h, w, 3)"
            )
        if (type(alpha) is not float and type(alpha) is not int) or alpha < 0:
            raise TypeError("alpha must be a non-negative number")
        if (type(beta) is not float and type(beta) is not int) or beta < 0:
            raise TypeError("beta must be a non-negative number")
        self.style_image = self.scale_image(style_image)
        self.content_image = self.scale_image(content_image)
        self.alpha = alpha
        self.beta = beta

    @staticmethod
    def scale_image(image):
        """Rescales the image such that its pixels values are between
        0 and 1 and its largest side is 512 pixels
        Arguments:
            image {np.ndarray} -- the image to be scaled
        Returns:
            np.ndarray -- the scaled image
        """
        if type(image) is not np.ndarray or image.ndim != 3:
            raise TypeError("image must be a np.ndarray with shape (h, w, 3)")
        h, w, _ = image.shape
        if h > w:
            h_new = 512
            w_new = int(w * (512 / h))
        else:
            w_new = 512
            h_new = int(h * (512 / w))
        image = image[tf.newaxis, ...]
        image = tf.image.resize(image, (h_new, w_new))
        image = image / 255
        image = tf.clip_by_value(image, 0, 1)
        return image
