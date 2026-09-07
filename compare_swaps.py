import cv2
import numpy as np

image_a = cv2.imread("SimSwap/raw_swap_A.jpg")
image_b = cv2.imread("SimSwap/raw_swap_B.jpg")

if image_a is None or image_b is None:
    raise FileNotFoundError("Could not load one of the raw swap images")

difference = np.abs(
    image_a.astype(np.float32)
    - image_b.astype(np.float32)
)

print("Mean pixel difference:", difference.mean())
print("Maximum pixel difference:", difference.max())