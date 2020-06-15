import os
import argparse
import numpy as np
import cv2

from dummy_datasets import get_ade150_dataset
from coco_format import *

def make_annotations(ann_dir, im_list):
    cm_dir = os.path.join(ann_dir, "cm")
    pm_dir = os.path.join(ann_dir, "pm")

    annotations = []
    for i, im_name in enumerate(im_list):
        print(i, im_name, len(annotations))

        cm_path = os.path.join(cm_dir, im_name.replace('.jpg', '.png'))
        pm_path = os.path.join(pm_dir, im_name.replace('.jpg', '.png'))
        cm = cv2.imread(cm_path, 0)
        pm = cv2.imread(pm_path, 0)
        if cm is None or pm is None:
            print("Skipping", im_name)
            continue

        pm = pm / 255
        bad = (pm < 0.5)

        for j in np.unique(cm):
            mask = (cm == j)
            mask[bad] = 0

            ann = make_ann(mask)
            ann["image_id"] = i + 1
            ann["category_id"] = int(j + 1)
            ann["id"] = len(annotations) + 1
            annotations.append(ann)
    return annotations

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-d', '--dataset', type=str, default="ade20k")
    parser.add_argument('-s', '--split', type=str, default="val")
    parser.add_argument('-o', '--out_dir', type=str, default="../data/ade20k/predictions/pspnet/")
    args = parser.parse_args()

    data_dir = "../data/{}/".format(args.dataset)
    im_dir = os.path.join(data_dir, "images")

    # Load im_list
    im_list = os.path.join(data_dir, "im_lists/{}.txt".format(args.split))
    with open(im_list,'r') as f:
        im_list = f.read().splitlines()

    # Load cat_list
    cat_list = get_ade150_dataset()

    annotations = make_annotations(args.out_dir, im_list)
    images = make_images(im_list, im_dir)
    categories = make_categories(cat_list)

    out_file = os.path.join(args.out_dir, "predictions.json")
    save_ann_fn(images, annotations, categories, out_file)





