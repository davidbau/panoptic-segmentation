import os
import argparse
import random

from pycocotools.coco import COCO


def split_bundles(coco, bundle_size=30):
    bundles = []
    for catId in coco.cats:
        annIds = coco.getAnnIds(catIds=[catId])
        anns = coco.loadAnns(annIds)
        random.shuffle(anns)

        splits = [anns[i:i + bundle_size] for i in range(0, len(anns), bundle_size)]
        for split in splits:
            bundle = make_bundle(coco, split)
            bundles.append(bundle)
    return bundles

def make_bundle(coco, anns):
    imgIds = set([ann["image_id"] for ann in anns])
    catIds = set([ann["category_id"] for ann in anns])
    images = [coco.imgs[id] for id in imgIds]
    categories = [coco.cats[id] for id in catIds]

    bundle = {}
    bundle["images"] = images
    bundle["annotations"] = anns
    bundle["categories"] = categories
    return bundle

def threshold(coco, t):
    anns = coco.dataset["annotations"]
    anns = [ann for ann in anns if ("score" not in ann) or ("score" in ann and ann["score"] >= t)]
    coco.dataset["annotations"] = anns
    coco.createIndex()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-f', '--ann_fn', type=str)
    args = parser.parse_args()

    coco = COCO(args.ann_fn)
