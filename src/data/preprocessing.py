"""Field-photo cleaning (metadata-strip crop, date-stamp inpaint) and left/right split."""
import cv2
import numpy as np


def clean_image_rgb(image, crop_bottom_fraction=0.10):
    # crops the metadata strip CSIRO burns into the bottom of field photos,
    # then inpaints the orange date-stamp that's sometimes burned in as well
    height = image.shape[0]
    crop_height = int(height * (1 - crop_bottom_fraction))
    image = image[:crop_height, :].copy()

    hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
    lower_orange = np.array([5, 150, 150])
    upper_orange = np.array([25, 255, 255])
    mask = cv2.inRange(hsv, lower_orange, upper_orange)
    mask = cv2.dilate(mask, np.ones((3, 3), np.uint8), iterations=2)
    if mask.sum() > 0:
        try:
            image = cv2.inpaint(image, mask, 3, cv2.INPAINT_TELEA)
        except cv2.error as error:
            print(f'WARNING inpaint failed, keeping cropped-only image: {error}')

    return image


def split_left_right(image):
    midpoint = image.shape[1] // 2
    return image[:, :midpoint].copy(), image[:, midpoint:].copy()


def load_and_preprocess_image(path, clean=True):
    raw = cv2.imread(path)
    if raw is None:
        print(f'WARNING could not read: {path}')
        return None
    rgb = cv2.cvtColor(raw, cv2.COLOR_BGR2RGB)
    return clean_image_rgb(rgb) if clean else rgb
