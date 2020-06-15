import os
import json
import numpy as np
# import logging
# import sys
# logging.basicConfig(stream=sys.stdout, level=logging.DEBUG)

from pycocotools.coco import COCO
from pycocotools import mask as COCOmask
from pycocotools.cocoeval import COCOeval

# from coco_utils.coco_format import *
# import json_dataset_evaluator

def print_stats(coco):
    print("{} images, {} annotations".format(len(coco.dataset["images"]), len(coco.dataset["annotations"])))
    counts = {}
    for cat in coco.cats:
        catName = coco.cats[cat]["name"]
        annIds = coco.getAnnIds(catIds=[cat])
        counts[cat] = len(annIds)
    print(counts)

def match_categories(cocoGt, cocoDt):
    cat_list = []
    for i in cocoDt.cats:
        cat_name = cocoDt.cats[i]["name"]
        for j in cocoGt.cats:
            if cat_name == cocoGt.cats[j]["name"]:
                cat_list.append(cat_name)
    print("Matched categories:", cat_list)
    return cat_list

def filter_categories(coco, cat_list):
    anns = coco.dataset["annotations"]
    matches = []
    for ann in anns:
        cat_id = ann["category_id"]
        cat_name = coco.cats[cat_id]["name"]
        if cat_name in cat_list:
            ann["category_id"] = cat_list.index(cat_name)
            matches.append(ann)
    print("Filter categories : {} -> {} annotations".format(len(anns), len(matches)))

    coco_new = COCO()
    coco_new.dataset = {}
    coco_new.dataset["images"] = coco.dataset["images"]
    coco_new.dataset["annotations"] = matches
    coco_new.dataset["categories"] = make_categories(cat_list)
    coco_new.createIndex()
    return coco_new

def prepare(coco):
    anns = coco.dataset["annotations"]
    filtered = [ann for ann in anns if "score" not in ann or ann["score"] > 0.2]
    print("Prepping : {} -> {} annotations".format(len(anns), len(filtered)))

    for ann in filtered:
        ann["iscrowd"] = 0
        ann["area"] = COCOmask.area(ann["segmentation"])

    coco_new = COCO()
    coco_new.dataset = {}
    coco_new.dataset["images"] = coco.dataset["images"]
    coco_new.dataset["annotations"] = filtered
    coco_new.dataset["categories"] = coco.dataset["categories"]
    coco_new.createIndex()
    return coco_new

def study(cocoGt, cocoDt, cat_list):
    cocoGt = filter_categories(cocoGt, cat_list)
    cocoDt = filter_categories(cocoDt, cat_list)

    coco_eval = COCOeval(cocoGt, cocoDt, iouType='segm')
    coco_eval.params.maxDets = [100]
    coco_eval.params.areaRng = ['all']
    coco_eval.evaluate()

    counts_gt = {}
    counts_dt = {}
    matches = {}
    for evalImg in coco_eval.evalImgs:
        if evalImg == None:
            continue
        img_id = evalImg['image_id']
        cat_id = evalImg['category_id']
        aRng = evalImg['aRng']
        maxDet = evalImg['maxDet']

        # print(img_id, cat_id, aRng, maxDet)
        if cat_id not in matches:
            counts_gt[cat_id] = 0
            counts_dt[cat_id] = 0
            matches[cat_id] = 0

        gtIds = evalImg['gtIds']
        dtIds = evalImg['dtIds']
        counts_gt[cat_id] += len(gtIds)
        counts_dt[cat_id] += len(dtIds)

        dtMatches = evalImg['dtMatches'][0] # IOU = 0.5
        for m in dtMatches:
            if m != 0:
                matches[cat_id] += 1

    for catId in counts_gt:
        catName = cocoGt.cats[catId]["name"]
        m = matches[catId]
        c_gt = counts_gt[catId]
        c_dt = counts_dt[catId]
        print(catName, m, 1.*m/c_dt, 1.*m/c_gt)

    # print("Counts GT:", counts_gt)
    # print("Counts DT:", counts_dt)
    # print("Matches:", matches)

    print_stats(cocoGt)
    print_stats(cocoDt)


if __name__ == "__main__":
    # gt_fn = "../data/ade20k_val_instances.json"
    # dt_fn = "../data/ade20k_val_predictions.json"
    gt_fn = "../data/ade20k/annotations/instances_ade20k_val.json"
    dt_fn = "../data/ade20k/predictions/maskrcnn_coco/amt/predictions.json"
    cocoGt = COCO(gt_fn)
    cocoDt = COCO(dt_fn)
    cocoGt = prepare(cocoGt)
    cocoDt = prepare(cocoDt)

    cat_list = match_categories(cocoGt, cocoDt)
    study(cocoGt, cocoDt, cat_list)

