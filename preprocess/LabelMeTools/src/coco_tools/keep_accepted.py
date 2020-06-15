import os
import sys
sys.path.append("../coco_utils")
import argparse

from pycocotools.coco import COCO
from coco_format import *

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-f', '--ann_fn', type=str, help='Annotation file')
    parser.add_argument('-o', '--out_fn', type=str, default=None, help='Output directory')
    args = parser.parse_args()
    if not args.out_dir:
        args.out_dir = args.ann_fn.replace(".json", "_accepted.json")
    print(args)

    coco = COCO(ann_fn)
    print_ann_fn(coco)

    annotations = [ann for ann in coco.dataset["annotations"] if ann["accepted"]]
    imgIds = set([ann["image_id"] for ann in annotations])
    images = [coco.imgs[imgId] for imgId in imgIds]
    catIds = set([ann["category_id"] for ann in annotations])
    categories = [coco.cats[catId] for catId in catIds]
    save_ann_fn(images, annotations, categories, out_fn)
    print_ann_fn(out_fn)
