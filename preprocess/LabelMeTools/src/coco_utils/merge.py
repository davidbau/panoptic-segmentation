import os
import argparse
import copy
from tqdm import tqdm

from pycocotools.coco import COCO
from coco_format import *

def merge_cocos(cocos):
    images = []
    annotations = []
    categories = []

    filename_to_id = {}
    catname_to_id = {}

    # For each coco
    for coco in tqdm(cocos):

        # For each img
        for img in coco.dataset["images"]:
            ann_ids = coco.getAnnIds(imgIds=[img["id"]])
            anns = coco.loadAnns(ann_ids)

            img = copy.deepcopy(img)
            img["id"] = len(images) + 1
            if img["file_name"] in filename_to_id:
                img["id"] = filename_to_id[img["file_name"]]
            else:
                filename_to_id[img["file_name"]] = img["id"]
                images.append(img)

            # For each ann
            for ann in anns:
                # Handle cat
                cat = coco.cats[ann["category_id"]]
                cat = copy.deepcopy(cat)
                cat["id"] = len(categories) + 1
                if cat["name"] in catname_to_id:
                    cat["id"] = catname_to_id[cat["name"]]
                else:
                    catname_to_id[cat["name"]] = cat["id"]
                    categories.append(cat)

                # Handle ann
                ann = copy.deepcopy(ann)
                ann["id"] = len(annotations) + 1
                ann["image_id"] = img["id"]
                ann["category_id"] = cat["id"]
                annotations.append(ann)

    coco = COCO()
    coco.dataset["images"] = images
    coco.dataset["annotations"] = annotations
    coco.dataset["categories"] = categories
    coco.createIndex()
    return coco

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-i', '--in_dir', type=str, required=True)
    parser.add_argument('-o', '--out_fn', type=str)
    args = parser.parse_args()
    if not args.out_fn:
        args.out_fn = os.path.normpath(args.in_dir) + ".json"
    print(args)

    cocos = []
    for filename in sorted(os.listdir(args.in_dir)):
        if ".json" == os.path.splitext(filename)[1]:
            print("Loading", filename)
            ann_fn = os.path.join(args.in_dir, filename)
            coco = COCO(ann_fn)
            cocos.append(coco)

    coco = merge_cocos(cocos)
    save_coco(coco, args.out_fn)

    # Verify out_fn
    coco = COCO(args.out_fn)
    print_coco(coco)

