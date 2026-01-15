"""Albumentations transform simulating a photo taken from farther back."""
import random

import cv2
import numpy as np
import albumentations as A


class SimulateCameraDistance(A.ImageOnlyTransform):
    """Shrinks the frame and pads the rest with black to imitate a phone shot
    taken from farther back.

    Adapted from the 1st place solution's scale+pad augmentation, rebuilt as
    a composable Albumentations transform instead of an inline pipeline step.
    """

    def __init__(self, scale_range=(0.85, 1.0), p=0.2):
        super().__init__(p=p)
        self.scale_range = scale_range

    def apply(self, img, **params):
        scale = random.uniform(*self.scale_range)
        height, width = img.shape[:2]
        resized = cv2.resize(img, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)
        resized_height, resized_width = resized.shape[:2]

        canvas = np.zeros_like(img)
        top = random.randint(0, height - resized_height)
        left = random.randint(0, width - resized_width)
        canvas[top:top + resized_height, left:left + resized_width] = resized
        return canvas

    def get_transform_init_args_names(self):
        return ('scale_range',)
