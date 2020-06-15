import os
import sys
sys.path.append("../coco_utils")
import argparse
from tqdm import tqdm

from pycocotools.coco import COCO
from pycocotools import mask as COCOmask
from pycocotools.cocoeval import COCOeval

from coco_format import *

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-f', '--ann_fn', type=str, help='Annotation file')
    parser.add_argument('-a', '--ann_list', type=str, help='List of annotations. The output of maskrcnn')
    parser.add_argument('-o', '--out_fn', type=str, default=None, help='Output coco file')
    args = parser.parse_args()
    if not args.out_fn:
        args.out_fn = args.ann_fn.replace(".json", "_pred.json")
    print(args)

    coco = COCO(args.ann_fn)
    ann_list = read_json(args.ann_list)

    coco.dataset["annotations"] = make_annotations(ann_list)
    save_coco(coco, args.out_fn)
    
    coco = COCO(args.out_fn)
    print_coco(coco)
