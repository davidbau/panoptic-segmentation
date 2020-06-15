import os
import argparse
from tqdm import tqdm

from pycocotools.coco import COCO
from pycocotools import mask as COCOmask

import sys
sys.path.append("../coco_utils")
from coco_format import *

ade20k_gt = "/Users/jeffreyhu/Documents/Torralba/datasets/ade20k/annotations/instances_train.json"
# splitB_gt = "/data/vision/torralba/ade20k-places/data/annotation/ade_challenge/ann_files/b_split.json"

class SimulatedAnnotator:

    def __init__(self):
        self.thresholdIOU = 0.8
        self.filenameToImg = {}
        self.nameToCat = {}

        self.cocoGt = COCO(ade20k_gt)
        self.setup(self.cocoGt)

    def setup(self, coco):
        for imgId in coco.imgs:
            img = coco.imgs[imgId]
            filename = img["file_name"]
            alternate = "ade_challenge/images/" + filename
            self.filenameToImg[filename] = img
            self.filenameToImg[alternate] = img

        for catId in coco.cats:
            cat = coco.cats[catId]
            self.nameToCat[cat["name"]] = cat

    def annotate(self, cocoDt):
        for imgId in cocoDt.imgs:
            for catId in cocoDt.cats:
                annIdsDt = cocoDt.getAnnIds(imgIds=[imgId], catIds=[catId])
                annsDt = cocoDt.loadAnns(annIdsDt)

                # cocoGt ids do not match cocoDt ids
                cocoGt = self.cocoGt
                imgIdGt = self.filenameToImg[cocoDt.imgs[imgId]["file_name"]]["id"]
                catIdGt = self.nameToCat[cocoDt.cats[catId]["name"]]["id"]

                annIdsGt = cocoGt.getAnnIds(imgIds=[imgIdGt], catIds=[catIdGt])
                annsGt = cocoGt.loadAnns(annIdsGt)

                gts = [ann["segmentation"] for ann in annsGt]
                dts = [ann["segmentation"] for ann in annsDt]
                iscrowds = [0 for _ in gts]
                if len(gts) == 0 or len(dts) == 0:
                    for annDt in annsDt:
                        annDt["completed_task_sim"] = {}
                        annDt["completed_task_sim"]["type"] = "yesno"
                        annDt["completed_task_sim"]["accepted"] = False
                        annDt["completed_task_sim"]["iou"] = 0
                    continue

                ious = COCOmask.iou(dts, gts, iscrowds)
                for iousDt, annDt in zip(ious, annsDt):
                    iou = np.max(iousDt)
                    # annGt = annsGt[np.argmax(iousDt)]
                    annDt["completed_task_sim"] = {}
                    annDt["completed_task_sim"]["type"] = "yesno"
                    annDt["completed_task_sim"]["accepted"] = float(iou) >= self.thresholdIOU
                    annDt["completed_task_sim"]["iou"] = iou

    # def annotate_old(self, cocoDt):
    #     passed = []
    #     for dt_imgId in tqdm(cocoDt.imgs):
    #         dt_img = coco.imgs[dt_imgId]
    #         dt_annIds = coco.getAnnIds(imgIds=[dt_imgId])
    #         dt_anns = coco.loadAnns(dt_annIds)
    #         for dt_ann in dt_anns:
    #             dt_cat = cocoDt.cats[dt_ann["category_id"]]
    #             dts = [dt_ann["segmentation"]]

    #             # Filter by imgId and catId
    #             gt_img = self.filenameToImg[dt_img["file_name"]]
    #             gt_cat = self.nameToCat[dt_cat["name"]]
    #             gt_annIds = self.cocoGt.getAnnIds(imgIds=[gt_img["id"]], catIds=[gt_cat["id"]])
    #             gt_anns = self.cocoGt.loadAnns(gt_annIds)
    #             gts = [ann["segmentation"] for ann in gt_anns]
    #             if len(gts) == 0:
    #                 continue

    #             iscrowds = [0 for _ in gts]
    #             ious = COCOmask.iou(dts, gts, iscrowds)

    #             # Check if max iou is greater than threshold
    #             max_dt_ious = np.max(ious, axis=1)
    #             max_dt_gtIds = np.argmax(ious, axis=1)
    #             for max_dt_iou, max_dt_gtId in zip(max_dt_ious, max_dt_gtIds):
    #                 if max_dt_iou >= self.thresholdIOU:
    #                     passed.append(dt_ann)
    #     return passed

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-f', '--ann_fn', type=str, required=True, help='Annotation file')
    parser.add_argument('-o', '--out_fn', type=str, default=None, help='Output coco file')
    args = parser.parse_args()
    if not args.out_fn:
        args.out_fn = args.ann_fn.replace(".json", "_annotated.json")
    print(args)

    coco = COCO(args.ann_fn)
    annotator = SimulatedAnnotator()
    annotator.annotate(coco)

    print_coco(coco)
    save_coco(coco, args.out_fn)
