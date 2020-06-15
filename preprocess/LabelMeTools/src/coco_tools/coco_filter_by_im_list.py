import os
import sys
sys.path.append("../coco_utils")
import argparse

from coco_utils.coco_format import *
from coco_utils.dummy_datasets import *

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-f', '--ann_fn', type=str, help='Annotation file')
    parser.add_argument('-i', '--im_list', type=str, help='List of images')
    parser.add_argument('-o', '--out_fn', type=str, default=None, help='Output coco file')
    args = parser.parse_args()
    if not args.out_fn:
        args.out_fn = args.ann_fn.replace(".json", "_filtered.json")
    print(args)

    coco = COCO(args.ann_fn)
    im_list = set(read_list(args.im_list))

    filtered_images = []
    filtered_annotations = []
    for img in images:
        if img["file_name"] in im_list:
            annIds = coco.getAnnIds(imgIds=[img["id"]])
            anns = coco.loadAnns(annIds)
            filtered_images.append(img)
            filtered_annotations.extend(anns)

    images = filtered_images
    annotations = filtered_annotations
    categories = coco.dataset["categories"]
    save_ann_fn(images, annotations, categories, args.out_fn)
    print_ann_fn(args.out_fn)
