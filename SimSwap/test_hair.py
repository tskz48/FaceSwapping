import cv2
import numpy as np

from hair_utils import get_hair_mask, load_hair_model


model = load_hair_model()

image = cv2.imread("../source1.jpg")

if image is None:
    raise ValueError("Could not load source image")

hair_mask = get_hair_mask(image, model)

source_hair = cv2.bitwise_and(
    image,
    image,
    mask=(hair_mask * 255).astype(np.uint8)
)

cv2.imwrite("../source_hair.jpg", source_hair)

cv2.imwrite(
    "../hair_mask.jpg",
    hair_mask * 255,
)

print("Saved hair_mask.jpg")