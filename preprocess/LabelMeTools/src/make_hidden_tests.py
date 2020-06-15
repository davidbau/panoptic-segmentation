import os
import argparse
import copy
from tqdm import tqdm

from pycocotools import mask as COCOmask
from pycocotools.coco import COCO

from coco_utils.coco_format import save_ann_fn

def print_stats(coco):
    print("{} images, {} annotations".format(len(coco.dataset["images"]), len(coco.dataset["annotations"])))

def make_hidden_tests(cocoGt, cocoDt):
    annotations = []
    for imgId in tqdm(cocoDt.imgs):
        annIdsGt = cocoGt.getAnnIds(imgIds=[imgId])
        annIdsDt = cocoDt.getAnnIds(imgIds=[imgId])
        annsGt = cocoGt.loadAnns(annIdsGt)
        annsDt = cocoDt.loadAnns(annIdsDt)
        gts = [ann["segmentation"] for ann in annsGt]
        dts = [ann["segmentation"] for ann in annsDt]
        iscrowds = [0 for _ in gts]
        if len(gts) == 0 or len(dts) == 0:
            continue

        ious = COCOmask.iou(dts, gts, iscrowds)
        for iousDt, annDt in zip(ious, annsDt):
            for iou, annGt in zip(iousDt, annsGt):
                if (0.2 < iou and iou < 0.5) or iou > 0.8:
                    annGt = copy.deepcopy(annGt)
                    annDt = copy.deepcopy(annDt)

                    ann = {}
                    ann["id"] = len(annotations) + 1
                    ann["image_id"] = annDt["image_id"]
                    ann["category_id"] = annGt["category_id"] # Use ground truth category
                    ann["segmentation"] = annDt["segmentation"]
                    ann["bbox"] = annDt["bbox"]
                    ann["area"] = annDt["area"]
                    ann["iscrowd"] = 0
                    ann["hidden_test"] = {}
                    ann["hidden_test"]["segmentation"] = annGt["segmentation"]
                    ann["hidden_test"]["bbox"] = annGt["bbox"]
                    ann["hidden_test"]["iou"] = iou
                    annotations.append(ann)

    imgIds = set([ann["image_id"] for ann in annotations])
    catIds = set([ann["category_id"] for ann in annotations])
    images = [cocoGt.imgs[imgId] for imgId in imgIds]
    categories = [cocoGt.cats[catId] for catId in catIds]

    coco = COCO()
    coco.dataset["images"] = images
    coco.dataset["annotations"] = annotations
    coco.dataset["categories"] = categories
    coco.createIndex()
    return coco

def verify_hidden_test(ann):
    dts = [ann["segmentation"]]
    gts = [ann["hidden_test"]["segmentation"]]
    iscrowds = [0 for _ in gts]
    ious = COCOmask.iou(dts, gts, iscrowds)
    print(ann["hidden_test"]["iou"], ious)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-gt', '--ground_truth', type=str, default="../../LabelMe-Lite/data/ade20k/annotations/instances_val.json")
    parser.add_argument('-dt', '--detections', type=str, default="../../LabelMe-Lite/data/ade20k/annotations/maskrcnna_val.json")
    args = parser.parse_args()

    cocoGt = COCO(args.ground_truth)
    cocoDt = COCO(args.detections)
    print_stats(cocoGt)
    print_stats(cocoDt)

    coco = make_hidden_tests(cocoGt, cocoDt)
    print_stats(cocoGt)

    out_file = "./hidden_test.json"
    images = coco.dataset["images"]
    annotations = coco.dataset["annotations"]
    categories = coco.dataset["categories"]
    save_ann_fn(images, annotations, categories, out_file)
