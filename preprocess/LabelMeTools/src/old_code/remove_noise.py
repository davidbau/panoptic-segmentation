import cv2
import numpy as np

def remove_noise(category_mask):
    kernel = np.ones((3,3),np.uint8) # Use odd kernel size

    background = (category_mask == 0)
    background = np.array(background, dtype=np.uint8)
    opened = cv2.morphologyEx(background, cv2.MORPH_OPEN, kernel)
    noise = (background != opened)

    # Claim in category order
    for i in np.unique(category_mask):
        if i == 0:
            continue
        cat_mask = np.array(category_mask == i, dtype=np.uint8)
        cat_mask = cv2.dilate(cat_mask, kernel)
        claimed = np.logical_and(cat_mask, noise)
        category_mask[claimed] = i
    return category_mask