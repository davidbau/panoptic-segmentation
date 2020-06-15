import os
import argparse
import numpy as np
import matplotlib.pyplot as plt

from pycocotools.coco import COCO

import sys
sys.path.append("../coco_utils")
from visualize import *

def get_examples(coco):
    anns = coco.dataset["annotations"]
    anns = [ann for ann in anns if ann["completed_task_sim"]["iou"] >= 0.9]

    anns_acc = [ann for ann in anns if not ann["completed_task"]["accepted"]]
    # anns_acc_with_agreement = filter_agreement(anns_acc, agreement=3)

    print(len(anns_acc))
    for ann in anns_acc:
        vis = visualize_ann(coco, ann)
        cv2.imwrite("plots/images/{}.jpg".format(ann["id"]), vis)

def filter_agreement(anns, agreement):
    counts = {}
    for ann in anns:
        key = str(ann["segmentation"])
        if key not in counts:
            counts[key] = 0
        counts[key] += 1

    anns = [ann for ann in anns if counts[str(ann["segmentation"])] >= agreement]
    return anns

def visualize_ann(coco, ann):
    im_dir = "../../data/ade20k/images"

    img = coco.imgs[ann["image_id"]]
    img["file_name"] = img["file_name"].replace("ade_challenge/images/", "")
    print(ann["id"], img["file_name"])
    image = load_image(coco, im_dir, img)
    vis = vis_ann(coco, image, ann, showClass=True, showBbox=False, showSegm=True, showKps=False, crop=True)
    return vis

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-f', '--ann_fn', type=str)
    args = parser.parse_args()

    coco = COCO(args.ann_fn)
    get_examples(coco)
