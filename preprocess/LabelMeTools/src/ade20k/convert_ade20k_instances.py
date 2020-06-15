import os
import argparse
import numpy as np
import cv2

from dummy_datasets import get_ade_dataset
from coco_format import *

def make_ade20k_annotations(ann_dir, im_list):
    ins_dir = os.path.join(ann_dir, "instances")

    annotations = []
    for i, im_name in enumerate(im_list):
        print(i, im_name, len(annotations))
        
        ann_path = os.path.join(ins_dir, im_name).replace('.jpg', '.png')
        ann_image = cv2.imread(ann_path)
        if ann_image is None:
            print("Skipping", ann_path)
            continue
        
        crowd_mask = ann_image[:,:,0]
        ins_mask = ann_image[:,:,1]
        cat_mask = ann_image[:,:,2]

        for ins in np.unique(ins_mask):
            if ins == 0:
                continue
            mask = (ins_mask == ins)
            cat = np.sum(cat_mask[mask]) / np.sum(mask)
            crowd = np.max(crowd_mask[mask])
            
            ann = make_ann(mask, iscrowd=crowd)
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
    cat_list = get_ade_dataset()

    annotations = make_ade20k_annotations(ann_dir, im_list)
    images = make_images(im_list, im_dir)
    categories = make_categories(cat_list)

    out_fn = os.path.join(ann_dir, "instances_{}.json".format(args.split))
    save_ann_fn(images, annotations, categories, out_fn)
    print_ann_fn(out_fn)


