import os
import argparse
import copy
from tqdm import tqdm

from pycocotools.coco import COCO
from labelmelite_client import LabelMeLiteClient

import sys
sys.path.append("../coco_utils")
from coco_format import *
from merge import merge_cocos

def post_process(results):
    for result in results:
        for ann in result["annotations"]:
            if "current_task" in ann:
                ann["completed_task"] = ann["current_task"]
                ann.pop('current_task', None)

def compile_results(results):
    post_process(results)

    coco = COCO()
    for result in results:
        coco_result = COCO()
        coco_result.dataset = result
        coco_result.createIndex()
        coco = merge_cocos(coco, coco_result)
    return coco

def filter_by_task(coco, task_type="yesno"):
    filtered = COCO()
    filtered.dataset = copy.deepcopy(coco.dataset)

    annotations = []
    for ann in filtered.dataset["annotations"]:
        task = ann["completed_task"]
        if task["type"] == task_type:
            annotations.append(ann)
    filtered.dataset["annotations"] = annotations
    filtered.createIndex()
    return filtered

def filter_accepted(coco, accepted=True):
    filtered = COCO()
    filtered.dataset = copy.deepcopy(coco.dataset)

    annotations = []
    for ann in filtered.dataset["annotations"]:
        task = ann["completed_task"]
        if task["accepted"] == accepted:
            annotations.append(ann)
    filtered.dataset["annotations"] = annotations
    filtered.createIndex()
    return filtered

def split_results_by_worker(results):
    split = {}
    for result in results:
        bundle_info = result["bundle_info"]
        worker_id = ""
        if "worker_id" in bundle_info:
            worker_id = bundle_info["worker_id"]

        if worker_id not in split:
            split[worker_id] = []
        split[worker_id].append(result)
    return split

def save_and_split_results(results, out_dir):
    coco = compile_results(results)
    save_coco(coco, os.path.join(out_dir, "all_results.json"))

    yesno = filter_by_task(coco, task_type="yesno")
    yesno_accepted = filter_accepted(yesno)
    yesno_rejected = filter_accepted(yesno, accepted=False)
    save_coco(yesno, os.path.join(out_dir, "yesno.json"))
    save_coco(yesno_accepted, os.path.join(out_dir, "yesno_accepted.json"))
    save_coco(yesno_rejected, os.path.join(out_dir, "yesno_rejected.json"))

    edit = filter_by_task(coco, task_type="edit")
    save_coco(edit, os.path.join(out_dir, "edit.json"))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-j', '--job_id', type=str, required=True)
    parser.add_argument('-l', '--local', action='store_true')
    args = parser.parse_args()
    print(args)

    lml_client = LabelMeLiteClient(args.local)
    results = lml_client.get_results(args.job_id)

    out_dir = os.path.join("output", args.job_id)
    save_and_split_results(results, out_dir)

    for worker_id, worker_results in split_results_by_worker(results).items():
        worker_out_dir = os.path.join(out_dir, worker_id)
        save_and_split_results(worker_results, worker_out_dir)
