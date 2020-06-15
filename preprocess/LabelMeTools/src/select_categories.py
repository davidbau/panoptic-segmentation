import os
import argparse
import copy
from tqdm import tqdm

from pycocotools import mask as COCOmask
from pycocotools.coco import COCO

from coco_utils.coco_format import *


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-f', '--ann_fn', type=str, required=True, help='Annotation file')
    parser.add_argument('-o', '--out_fn', type=str, default=None, help='Output coco file')
    args = parser.parse_args()
    if not args.out_fn:
        args.out_fn = args.ann_fn.replace(".json", "_5random.json")
    print(args)

    coco = COCO(args.ann_fn)
    select_cats = ["car", "door", "counter", "traffic light", "flag"]
    catIds = [cat["id"] for cat in coco.dataset["categories"] if cat["name"] in select_cats]
    numPerClass = 30*3

    annotations = []
    for catId in catIds:
    	annIds = coco.getAnnIds(catIds=[catId])
    	anns = coco.loadAnns(annIds)[:numPerClass]
    	annotations.extend(anns)

    imgIds = set([ann["image_id"] for ann in annotations])
    catIds = set([ann["category_id"] for ann in annotations])
    images = [coco.imgs[imgId] for imgId in imgIds]
    categories = [coco.cats[catId] for catId in catIds]
    save_ann_fn(images, annotations, categories, args.out_fn)
    print_ann_fn(args.out_fn)
