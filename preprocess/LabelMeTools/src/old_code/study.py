import os
import json
import logging
import numpy as np

from pycocotools.coco import COCO
from pycocotools import mask as COCOmask
from pycocotools.cocoeval import COCOeval

from coco_utils.coco_format import *

logger = logging.getLogger(__name__)

def print_stats(coco):
    print("{} images, {} annotations".format(len(coco.dataset["images"]), len(coco.dataset["annotations"])))
    counts = {}
    for cat in coco.cats:
        catName = coco.cats[cat]["name"]
        annIds = coco.getAnnIds(catIds=[cat])
        counts[catName] = len(annIds)
    print(counts)

def get_cat_list(coco):
    cat_list = [entry["name"] for entry in coco.dataset["categories"]]
    return cat_list

def get_cat_id(coco, cat):
    for entry in coco.dataset["categories"]:
        if entry["name"] == cat:
            return entry["id"]
    return None

def filter_categories(coco, cat_list):
    catToAnns = {}
    for cat in cat_list:
        catId = get_cat_id(coco, cat)
        if catId == None:
            continue

        annIds = coco.getAnnIds(catIds=[catId])
        catToAnns[cat] = coco.loadAnns(ids=annIds)

    annotations = []
    categories = []
    for i, cat in enumerate(cat_list):
        anns = catToAnns[cat]
        for ann in anns:
            ann["category_id"] = i+1
        annotations.extend(anns)
        categories.append({"id": i+1, "name": cat})

    coco.dataset["annotations"] = annotations
    coco.dataset["categories"] = categories
    coco.createIndex()

def threshold(coco, t):
    annotations = []
    for ann in coco.dataset["annotations"]:
        if ann["score"] >= t:
            annotations.append(ann)

    coco.dataset["annotations"] = annotations
    coco.createIndex()

def match(cocoGt, cocoDt):
    coco_eval = COCOeval(cocoGt, cocoDt, iouType='segm')
    coco_eval.params.maxDets = [100]
    coco_eval.params.areaRng = [[0 ** 2, 1e5 ** 2]]
    coco_eval.params.areaRngLbl = ['all']
    coco_eval.evaluate()

    gt_matched = []
    dt_matched = []
    for evalImg in coco_eval.evalImgs:
        if evalImg == None:
            continue
        img_id = evalImg['image_id']
        cat_id = evalImg['category_id']
        aRng = evalImg['aRng']
        maxDet = evalImg['maxDet']
        # print(img_id, cat_id, aRng, maxDet)
        # gtIds = evalImg['gtIds']
        # dtIds = evalImg['dtIds']

        gtMatches = evalImg['gtMatches'][0] # IOU = 0.5
        dtMatches = evalImg['dtMatches'][0] # IOU = 0.5
        for gtId in dtMatches:
            if gtId != 0:
                gt_matched.append(gtId)
        for dtId in gtMatches:
            if dtId != 0:
                dt_matched.append(dtId)
    return gt_matched, dt_matched

def study(cocoGt, cocoDt):
    cat_list0 = get_cat_list(cocoGt)
    cat_list1 = get_cat_list(cocoDt)
    cat_list = [c for c in cat_list0 if c in cat_list1]
    # cat_list = ['person']
    print(cat_list)

    filter_categories(cocoGt, cat_list)
    filter_categories(cocoDt, cat_list)
    threshold(cocoDt, 0.5)
    print_stats(cocoGt)
    print_stats(cocoDt)

    gt_matched, dt_matched = match(cocoGt, cocoDt)
    print(len(gt_matched))
    print(len(dt_matched))


def _log_detection_eval_metrics(category_list, coco_eval):
    def _get_thr_ind(coco_eval, thr):
        ind = np.where((coco_eval.params.iouThrs > thr - 1e-5) &
                       (coco_eval.params.iouThrs < thr + 1e-5))[0][0]
        iou_thr = coco_eval.params.iouThrs[ind]
        assert np.isclose(iou_thr, thr)
        return ind

    IoU_lo_thresh = 0.5
    IoU_hi_thresh = 0.95
    ind_lo = _get_thr_ind(coco_eval, IoU_lo_thresh)
    ind_hi = _get_thr_ind(coco_eval, IoU_hi_thresh)
    # precision has dims (iou, recall, cls, area range, max dets)
    # area range index 0: all area ranges
    # max dets index 2: 100 per image
    precision = coco_eval.eval['precision'][ind_lo:(ind_hi + 1), :, :, 0, 2]
    ap_default = np.mean(precision[precision > -1])
    logger.info(
        '~~~~ Mean and per-category AP @ IoU=[{:.2f},{:.2f}] ~~~~'.format(
            IoU_lo_thresh, IoU_hi_thresh))
    logger.info('{:.1f}'.format(100 * ap_default))
    for cls_ind, cls in enumerate(category_list):
        if cls == '__background__':
            continue
        # minus 1 because of __background__
        precision = coco_eval.eval['precision'][
            ind_lo:(ind_hi + 1), :, cls_ind - 1, 0, 2]
        ap = np.mean(precision[precision > -1])
        logger.info('{:.1f}'.format(100 * ap))
    logger.info('~~~~ Summary metrics ~~~~')
    coco_eval.summarize()


if __name__ == "__main__":
    gt_fn = "../../LabelMe-Lite/data/ade20k/instances_val.json"
    dt_fn = "../../LabelMe-Lite/data/ade20k/maskrcnnc_val.json"
    cocoGt = COCO(gt_fn)
    cocoDt = COCO(dt_fn)
    
    study(cocoGt, cocoDt)

