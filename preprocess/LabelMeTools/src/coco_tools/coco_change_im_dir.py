import os
import sys
sys.path.append("../coco_utils")
import argparse
import json
import logging
import numpy as np
from tqdm import tqdm

from pycocotools.coco import COCO
from pycocotools import mask as COCOmask
from pycocotools.cocoeval import COCOeval

from coco_utils.coco_format import *

def change_im_dir(coco, old_im_dir, new_im_dir):
    for imgId in tqdm(coco.imgs):
        img = coco.imgs[imgId]
        im_name = img["file_name"]
        full_path = os.path.join(old_im_dir, im_name)
        img["file_name"] = os.path.relpath(full_path, new_im_dir)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-f', '--ann_fn', type=str, help='Annotation file')

    parser.add_argument('-n', '--new_im_dir', type=str, help='New image directory')
    parser.add_argument('-d', '--old_im_dir', type=str, default="/data/vision/torralba/ade20k-places/data", help='Images directory')
    args = parser.parse_args()
    print(args)

    coco = COCO(args.ann_fn)
    out_fn = args.ann_fn.replace(".json", "_fixed.json")

    change_im_dir(coco, args.old_im_dir, args.new_im_dir)

    images = coco.dataset["images"]
    annotations = coco.dataset["annotations"]
    categories = coco.dataset["categories"]
    save_ann_fn(images, annotations, categories, args.out_fn)
