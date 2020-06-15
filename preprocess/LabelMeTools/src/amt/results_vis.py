import os
import argparse
import numpy as np
import copy

from pycocotools.coco import COCO

import sys
sys.path.append("../coco_utils")
from coco_format import *
from visualize import *

def visualize_result(result_fn, im_dir):
    coco = COCO(result_fn)
    out_dir = result_fn.replace(".json", "")
    print_coco(coco)
    print_times(coco)
    vis_coco(coco, im_dir, out_dir)

def print_times(coco):
    stats = {}
    stats["times"] = []
    for ann in coco.dataset["annotations"]:
        task = ann["completed_task"]
        stats["times"].append(task["annotationTime"])

    stats["avg_time"] = np.mean(stats["times"])
    stats["tot_time"] = np.sum(stats["times"])
    stats.pop("times", None)
    print(stats)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-j', '--job_id', type=str, required=True)
    parser.add_argument('-d', '--im_dir', type=str, default="/data/vision/torralba/ade20k-places/data", help='Images directory')
    args = parser.parse_args()

    out_dir = os.path.join("output", args.job_id)
    for root, dirs, files in os.walk(out_dir):
        for file in files:
            if ".json" in file:
                result_fn = os.path.join(root, file)
                print(result_fn)
                visualize_result(result_fn, args.im_dir)
