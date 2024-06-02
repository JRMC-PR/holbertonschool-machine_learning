#!/usr/bin/env python3
"""
Module to perform a convolution on grayscale
images with custom padding and stride.
"""
import numpy as np


def convolve_grayscale(images, kernel, padding='same',
                       stride=(1, 1)):
    """
    Perform a convolution on grayscale images
    with custom padding and stride.

    Parameters:
    images (numpy.ndarray): The grayscale images
    with shape (m, h, w).
    kernel (numpy.ndarray): The kernel for the
    convolution with shape (kh, kw).
    padding (str or tuple): The padding for the
    height and width of the image.
    stride (tuple): The stride for the height and
    width of the image.

    Returns:
    A numpy.ndarray containing the convolved images.
    """
    m, h, w = images.shape
    kh, kw = kernel.shape
    sh, sw = stride

    if isinstance(padding, tuple):
        ph, pw = padding
    elif padding == 'same':
        ph = max((kh - 1) // 2, kh // 2)
        pw = max((kw - 1) // 2, kw // 2)
    else:  # padding == 'valid'
        ph = pw = 0

    images_padded = np.pad(images, ((0, 0), (ph, ph), (pw, pw)), 'constant')
    output_h = (h - kh + 2 * ph) // sh + 1
    output_w = (w - kw + 2 * pw) // sw + 1
    output = np.zeros((m, output_h, output_w))

    for x in range(0, output_w * sw, sw):
        for y in range(0, output_h * sh, sh):
            output[:, y // sh, x // sw] = \
            (images_padded[:, y: y + kh, x: x + kw] * kernel).sum(axis=(1, 2))

    return output
