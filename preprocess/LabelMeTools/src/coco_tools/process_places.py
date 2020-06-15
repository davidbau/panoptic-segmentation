import os
import argparse
from pycocotools.coco import COCO

import sys
sys.path.append("../coco_utils")
from coco_format import *
from split_categories import *

def process_split(places_dir, split_fn):
    print("Processing Places split", split_fn)
    ann_fn = os.path.join(places_dir, "train_files/iteration0/predictions/splits/", os.path.basename(split_fn))

    # Save split predictions
    coco = COCO(split_fn)
    segm_list = os.path.join(places_dir, "train_files/iteration0/inference/annotation/places_challenge/ann_files/{}/segm.json".format(os.path.basename(split_fn)))
    ann_list = read_json(segm_list)
    coco.dataset["annotations"] = make_annotations(ann_list)
    save_coco(coco, ann_fn)
    print_coco(coco)

    # Split categories
    cat_dir = os.path.join(places_dir, "train_files/iteration0/predictions/categories/splits")
    split_categories(ann_fn, cat_dir)

def process_places(places_dir):
    ann_files_dir = os.path.join(places_dir, "ann_files")
    for i, filename in enumerate(sorted(os.listdir(ann_files_dir))):
        if "split" not in filename:
            continue
        split_fn = os.path.join(ann_files_dir, filename)
        process_split(places_dir, split_fn)

if __name__ == "__main__":
    places_dir = "/data/vision/torralba/ade20k-places/data/annotation/places_challenge"
    process_places(places_dir)
