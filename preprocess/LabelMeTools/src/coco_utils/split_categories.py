import os
import argparse

from pycocotools.coco import COCO

from coco_format import *

def split_categories(ann_fn, out_dir):
    coco = COCO(ann_fn)
    for cat_id in coco.cats:
        cat = coco.cats[cat_id]
        cat_fn = os.path.join(out_dir, cat["name"], os.path.basename(ann_fn))

        categories = [cat]
        annotations = [ann for ann in coco.dataset["annotations"] if ann["category_id"] == cat_id]
        img_ids = set([ann["image_id"] for ann in annotations])
        images = [coco.imgs[img_id] for img_id in img_ids]

        coco_cat = COCO()
        coco_cat.dataset["images"] = images
        coco_cat.dataset["annotations"] = annotations
        coco_cat.dataset["categories"] = categories
        save_coco(coco_cat, cat_fn)
        print_coco(coco_cat)
