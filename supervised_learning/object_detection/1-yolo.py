#!/usr/bin/env python3
"""This module contains the Yolo class
that uses the Yolo v3 algorithm to perform object detection
includes processing output method
"""
import numpy as np
import tensorflow as tf
import tensorflow.keras as K
from tensorflow.keras.activations import sigmoid  # type: ignore


class Yolo:
    """This class uses the Yolo v3 algorithm to perform object detection"""

    def __init__(self, model_path, classes_path, class_t, nms_t, anchors):
        """Yolo class constructor
        Args:
            model_path: is the path to where a Darknet Keras model is stored
            classes_path: is the path to where the list of class names used
                for the Darknet model, listed in order of index, can be found
            class_t: is a float representing the box score threshold for the
                initial filtering step
            nms_t: is a float representing the IOU threshold for non-max
                suppression
            anchors: is a numpy.ndarray of shape (outputs, anchor_boxes, 2)
                containing all of the anchor boxes:
                outputs: is the number of outputs (predictions) made by the
                    Darknet model
                anchor_boxes: is the number of anchor boxes used for each
                    prediction
                2: [anchor_box_width, anchor_box_height]
        """
        self.model = K.models.load_model(model_path)
        # Open file and read content
        with open(classes_path, "r") as f:
            self.class_names = f.read().splitlines()
        self.class_t = class_t
        self.nms_t = nms_t
        self.anchors = anchors

    def process_outputs(self, outputs, image_size):
        """Process Darknet outputs
        Args:
            outputs: list of numpy.ndarrays containing the predictions from the
                Darknet model for a single image:
                Each output will have the shape (grid_height, grid_width,
                anchor_boxes, 4 + 1 + classes)
                    grid_height & grid_width: the height and width of the
                    grid used
                        for the output
                    anchor_boxes: the number of anchor boxes used
                    4: (t_x, t_y, t_w, t_h)
                    1: box_confidence
                    classes: class probabilities for all classes
            image_size: numpy.ndarray containing the image’s original size
                [image_height, image_width]
        Returns: tuple of (boxes, box_confidences, box_class_probs):
            boxes: list of numpy.ndarrays of shape (grid_height,
            grid_width,
                anchor_boxes, 4) containing the processed boundary boxes
                for each
                output, respectively:
                4: (x1, y1, x2, y2)
                (x1, y1, x2, y2) should represent the boundary box relative to
                    original image
            box_confidences: list of numpy.ndarrays of shape (grid_height,
            grid_width,
                anchor_boxes, 1) containing the box confidences for
                each output,
                respectively
            box_class_probs: list of numpy.ndarrays of shape (grid_height,
            grid_width,
                anchor_boxes, classes) containing the box’s class
                probabilities for
                each output, respectively
        """
        boxes = []
        box_confidences = []
        box_class_probs = []

        # unpack the output
        for output in outputs:
            grid_height, grid_width, anchor_boxes, _ = output.shape
            box = np.zeros((grid_height, grid_width, anchor_boxes, 4))

            # Calculate the cx, cy, width and height of the boxes
            box_xy = sigmoid(output[..., :2])
            box_wh = tf.exp(output[..., 2:4])
            box_confidence = sigmoid(output[..., 4:5])
            box_class_prob = sigmoid(output[..., 5:])

            # Calculate the top left and bottom right corners of the boxes
            box[:, :, :, 0:2] = box_xy - (box_wh / 2.0)  # x1, y1
            box[:, :, :, 2:4] = box_xy + (box_wh / 2.0)  # x2, y2

            # Adjust boxes to be relative to the original image size
            box[:, :, :, 0:4] *= np.tile(image_size, 2) / np.tile(
                [grid_width, grid_height], 2)

            boxes.append(box)
            box_confidences.append(box_confidence)
            box_class_probs.append(box_class_prob)

        return boxes, box_confidences, box_class_probs
