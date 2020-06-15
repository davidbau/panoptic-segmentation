import os
import argparse
import numpy as np
import copy

from pycocotools.coco import COCO

import sys
sys.path.append("../coco_utils")
from coco_format import *
from annotator_sim import SimulatedAnnotator

annotator = SimulatedAnnotator()

def simulate_result(result_fn):
    coco = COCO(result_fn)
    annotator.annotate(coco)
    save_coco(coco, result_fn)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-j', '--job_id', type=str, required=True)
    args = parser.parse_args()

    out_dir = os.path.join("output", args.job_id)
    for root, dirs, files in os.walk(out_dir):
        for file in files:
            if ".json" in file:
                result_fn = os.path.join(root, file)
                print(result_fn)
                simulate_result(result_fn)
