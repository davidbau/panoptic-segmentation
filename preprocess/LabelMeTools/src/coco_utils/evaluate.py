import os
import argparse
import json
import logging
from collections import OrderedDict
from tqdm import tqdm

from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
from coco_format import *

logger = logging.getLogger("maskrcnn_benchmark.inference")

def do_coco_evaluation(coco_gt, coco_dt, iou_type="segm"):
    results = COCOResults("segm")
    logger.info("Evaluating predictions")

    coco_eval = COCOeval(coco_gt, coco_dt, iou_type)
    coco_eval.evaluate()
    coco_eval.accumulate()
    coco_eval.summarize()

    results.update(coco_eval)
    logger.info(results)
    return results


class COCOResults(object):
    METRICS = {
        "bbox": ["AP", "AP50", "AP75", "APs", "APm", "APl"],
        "segm": ["AP", "AP50", "AP75", "APs", "APm", "APl"],
        "box_proposal": [
            "AR@100",
            "ARs@100",
            "ARm@100",
            "ARl@100",
            "AR@1000",
            "ARs@1000",
            "ARm@1000",
            "ARl@1000",
        ],
        "keypoints": ["AP", "AP50", "AP75", "APm", "APl"],
    }

    def __init__(self, *iou_types):
        allowed_types = ("box_proposal", "bbox", "segm", "keypoints")
        assert all(iou_type in allowed_types for iou_type in iou_types)
        results = OrderedDict()
        for iou_type in iou_types:
            results[iou_type] = OrderedDict(
                [(metric, -1) for metric in COCOResults.METRICS[iou_type]]
            )
        self.results = results

    def update(self, coco_eval):
        if coco_eval is None:
            return
        from pycocotools.cocoeval import COCOeval

        assert isinstance(coco_eval, COCOeval)
        s = coco_eval.stats
        iou_type = coco_eval.params.iouType
        res = self.results[iou_type]
        metrics = COCOResults.METRICS[iou_type]
        for idx, metric in enumerate(metrics):
            res[metric] = s[idx]

    def __repr__(self):
        # TODO make it pretty
        return repr(self.results)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-g', '--coco_gt', type=str)
    parser.add_argument('-d', '--coco_dt', type=str)
    parser.add_argument('-c', '--category_id', type=int)
    args = parser.parse_args()

    coco_gt = COCO(args.coco_gt)
    coco_dt = COCO(args.coco_dt)
    print_ann_fn(coco_gt)
    print_ann_fn(coco_dt)

    if args.category_id:
        coco_gt = filter_category(coco_gt, args.category_id)
        coco_dt = filter_category(coco_dt, args.category_id)
        print_ann_fn(coco_gt)
        print_ann_fn(coco_dt)
    do_coco_evaluation(coco_gt, coco_dt)

