#!/usr/bin/env python3
"""YOLO module"""

from tensorflow import keras as K
import numpy as np


class Yolo:
    """yolo class algorithm"""
    def __init__(self, model_path, classes_path,
                 class_t, nms_t, anchors):
        """init function"""
        self.model = keras.models.load_model(model_path)
        with open(classes_path, 'r') as f:
            self.class_names = [line.strip() for line in f]
        self.class_t = class_t
        self.nms_t = nms_t
        self.anchors = anchors

    def sigmoid(self, z):
        """sigmoid"""
        return (1 / (1 + np.exp(-z)))

    def process_outputs(self, outputs, image_size):
        """process outputs function"""
        boxes = []
        box_confidences = []
        box_class_probs = []
        H, W = image_size[0], image_size[1]
        for i, grid in enumerate(outputs):
            tx = grid[..., 0]
            ty = grid[..., 1]
            tw = grid[..., 2]
            th = grid[..., 3]
            grid_w, grid_h = grid.shape[:2]
            detections_w = self.anchors[..., 0]
            detections_h = self.anchors[..., 1]
            detection_w = np.tile(detections_w[i], grid_w)
            detection_w = detection_w.reshape(grid_w, 1, len(detections_w[i]))
            detection_h = np.tile(detections_h[i], grid_h)
            detection_h = detection_h.reshape(grid_h, 1, len(detections_h[i]))
            cox = np.tile(np.arange(grid_w), grid_h)
            cox = cox.reshape(grid_w, grid_w, 1)
            coy = np.tile(np.arange(grid_h), grid_h)
            coy = coy.reshape(grid_h, grid_h).T
            coy = coy.reshape(grid_h, grid_h, 1)
            pred_x = (1 / (1 + np.exp(-tx))) + cox
            pred_y = (1 / (1 + np.exp(-ty))) + coy
            pred_w = np.exp(tw) * detection_w
            pred_h = np.exp(th) * detection_h
            pred_x /= grid_w
            pred_y /= grid_h
            pred_w /= self.model.input.shape[1]
            pred_h /= self.model.input.shape[2]
            x = (pred_x - (pred_w / 2)) * W
            y = (pred_y - (pred_h / 2)) * H
            x1 = (pred_x + (pred_w / 2)) * W
            y1 = (pred_y + (pred_h / 2)) * H
            box = np.zeros_like(grid[..., :4])
            box[..., 0] = x
            box[..., 1] = y
            box[..., 2] = x1
            box[..., 3] = y1
            boxes.append(box)
            box_confidences.append(np.expand_dims(
                sigmoid(grid[..., 4]), axis=-1))
            conf = grid[..., 5:]
            box_class_probs.append(sigmoid(conf))
        return (boxes, box_confidences, box_class_probs)
