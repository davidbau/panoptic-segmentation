import os
import sys
sys.path.append("../coco_utils")
import argparse

from coco_utils.coco_format import *
from coco_utils.dummy_datasets import *

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-i', '--im_list', type=str, help='List of images')
    parser.add_argument('-a', '--ann_list', type=str, help='List of annotations. Typically the output of maskrcnn')
    parser.add_argument('-c', '--cat_list', type=str, help='List of categories')

    parser.add_argument('-d', '--im_dir', type=str, default="/data/vision/torralba/ade20k-places/data", help='Images directory')
    parser.add_argument('-o', '--out_fn', type=str, default=None, help='Output coco file')
    args = parser.parse_args()
    if not args.out_fn:
        args.out_fn = args.im_list.replace(".txt", ".json")
    print(args)

    # Make categories
    categories = []
    if args.cat_list:
        cat_list = []
        if args.cat_list == "coco":
            cat_list = get_coco_dataset()
        elif args.cat_list == "ade100":
            cat_list = get_ade100_dataset()
        elif args.cat_list == "ade150":
            cat_list = get_ade150_dataset()
        else:
            cat_list = read_list(args.cat_list)

        categories = make_categories(cat_list)

    # Make annotations
    annotations = []
    if args.ann_list:
        ann_list = read_json(args.ann_list)
        annotations = make_annotations(ann_list)

    # Make images
    images = []
    if args.im_list:
        im_list = read_list(args.im_list)
        images = make_images(im_list, args.im_dir)

    save_ann_fn(images, annotations, categories, args.out_fn)
    print_ann_fn(args.out_fn)
