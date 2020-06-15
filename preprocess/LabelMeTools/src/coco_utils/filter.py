import os
import argparse

from pycocotools.coco import COCO

def filter_category(coco, catId):
    annIds = coco.getAnnIds(catIds=[catId])
    anns = coco.loadAnns(annIds)

    coco.dataset["annotations"] = anns
    coco.dataset["categories"] = [coco.cats[catId]]
    coco.createIndex()
    return coco
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-c', '--category_id', type=int)
    args = parser.parse_args()