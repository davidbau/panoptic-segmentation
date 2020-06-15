import os
import cv2
import json
import argparse
import re
import numpy as np
from tqdm import tqdm

from pycocotools import mask as COCOmask
from pycocotools.coco import COCO

def make_images(im_list, im_dir):
    images = []
    for im_name in tqdm(im_list):
        img = {}
        img["file_name"] = im_name
        img["id"] = len(images) + 1

        im_path = os.path.join(im_dir, im_name)
        im = cv2.imread(im_path)
        img["height"] = im.shape[0]
        img["width"] = im.shape[1]
        
        images.append(img)
    return images

def make_categories(cat_list):
    categories = []
    cat_list.remove("__background__")
    for name in cat_list:
        categories.append({"id": len(categories) + 1, "name": name})
    return categories

def make_annotations(ann_list):
    annotations = []
    for a in tqdm(ann_list):
        ann = {}
        ann["id"] = len(annotations) + 1
        ann["category_id"] = a["category_id"]
        ann["image_id"] = a["image_id"]
        ann["iscrowd"] = 0
        if "score" in a:
            ann["score"] = a["score"]

        segm = a["segmentation"]
        ann["segmentation"] = segm
        ann["area"] = int(COCOmask.area(segm))
        ann["bbox"] = list(COCOmask.toBbox(segm))
        annotations.append(ann)
    return annotations

def make_ann(mask, iscrowd=0):
    mask = np.asfortranarray(mask)
    mask = mask.astype(np.uint8)
    segm = COCOmask.encode(mask)
    segm["counts"] = segm["counts"].decode('ascii')

    ann = {}
    ann["segmentation"] = segm
    ann["bbox"] = list(COCOmask.toBbox(segm))
    ann["area"] = int(COCOmask.area(segm))
    ann["iscrowd"] = int(iscrowd)
    return ann

def save_coco(coco, out_fn, indent=2):
    dirname = os.path.dirname(out_fn)
    if dirname and not os.path.exists(dirname):
        os.makedirs(dirname)

    with open(out_fn, 'w') as f:
        json.dump(coco.dataset, f, indent=indent)
    print("Saved coco to", out_fn)

def print_coco(coco):
    images = coco.dataset["images"]
    annotations = coco.dataset["annotations"]
    categories = coco.dataset["categories"]
    if len(coco.imgs) == 0:
        coco.createIndex()

    # Statistics
    stats = {}
    stats["images"] = len(images)
    stats["annotations"] = len(annotations)
    stats["categories"] = len(categories)
    stats["images_with_annotations"] = len(set([ann["image_id"] for ann in annotations]))
    print("Statistics:", stats)

    # Count category frequencies
    counts = {}
    for cat in categories:
        ann_ids = coco.getAnnIds(catIds=[cat["id"]])
        counts[cat["name"]] = len(ann_ids)
    print("Category counts:", counts)

def print_coco_examples(coco, num_examples=3):
    # Show examples for each field
    for field in coco.dataset:
        print("{}".format(field))
        for item in coco.dataset[field][:num_examples]:
            print(item)

def read_json(file_name):
    with open(file_name, 'r') as f:
        return json.load(f)

def read_list(file_name):
    with open(file_name, 'r') as f:
        return f.read().splitlines()

def write_list(file_name, l):
    with open(file_name, 'w') as f:
        for item in l:
            f.write(item + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-f', '--ann_fn', type=str)
    args = parser.parse_args()

    coco = COCO(args.ann_fn)
    print_coco(coco)
    print_coco_examples(coco)

