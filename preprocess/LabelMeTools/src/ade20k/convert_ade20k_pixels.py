import os
import argparse
import numpy as np
import cv2

from dummy_datasets import get_ade150_dataset
from coco_format import *

def make_ade20k_annotations(ann_dir, im_list):
    pixel_dir = os.path.join(ann_dir, "pixels")

    annotations = []
    for i, im_name in enumerate(im_list):
        print(i, im_name, len(annotations))

        ann_path = os.path.join(pixel_dir, im_name).replace('.jpg', '.png')
        cat_mask = cv2.imread(ann_path, 0)
        if cat_mask is None:
            print("Skipping", ann_path)
            continue
        
        for cat in np.unique(cat_mask):
            if cat == 0:
                continue
            mask = (cat_mask == cat)

            ann = make_ann(mask)
            ann["image_id"] = i + 1
            ann["category_id"] = int(cat)
            ann["id"] = len(annotations) + 1
            annotations.append(ann)
    return annotations

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-s', '--split', type=str, default="val")
    args = parser.parse_args()

    data_dir = "../data/ade20k"
    im_dir = os.path.join(data_dir, "images")
    ann_dir = os.path.join(data_dir, "annotations")

    # Load im_list
    im_list_fn = os.path.join(data_dir, "im_lists/{}.txt".format(args.split))
    im_list = read_list(im_list_fn)

    # Load cat_list
    cat_list = get_ade150_dataset()

    annotations = make_ade20k_annotations(ann_dir, im_list)
    images = make_images(im_list, im_dir)
    categories = make_categories(cat_list)

    out_fn = os.path.join(ann_dir, "pixels_{}.json".format(args.split))
    save_ann_fn(images, annotations, categories, out_fn)
    print_ann_fn(out_fn)


