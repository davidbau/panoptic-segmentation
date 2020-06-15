import argparse
import os
import json
import math
import random

from pycocotools.coco import COCO

from dummy_dataset import *
from coco_format import save_coco, print_coco

def get_categories(dataset_name):
    cat_list = []
    if dataset_name == "coco":
        cat_list = get_coco_dataset()
    if dataset_name == "ade100":
        cat_list = get_ade100_dataset()
    if dataset_name == "ade150":
        cat_list = get_ade150_dataset()
    return make_categories(cat_list)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-f', '--ann_fn', type=str)
    parser.add_argument('-o', '--out_fn', type=str)
    parser.add_argument('-d', '--dataset', type=str)
    args = parser.parse_args()
    if not args.out_fn:
        args.out_fn = args.ann_fn.replace(".json", "_{}.json".format(args.dataset))
    print(args)

    coco = COCO(args.ann_fn)
    coco.dataset["categories"] = get_categories(args.dataset)

    save_coco(c, out_fn)

    # Verify out_fn
    coco = COCO(args.out_fn)
    print_coco(coco)