import os
import argparse
import numpy as np
import matplotlib.pyplot as plt

from pycocotools.coco import COCO

def plot_histogram(coco, cat_id=None):
    anns = coco.dataset["annotations"]
    anns = [ann for ann in anns if ann["completed_task_sim"]["iou"] != 0]
    if cat_id:
        anns = [ann for ann in anns if ann["category_id"] == cat_id]

    anns_acc = [ann for ann in anns if ann["completed_task"]["accepted"]]
    anns_acc_with_agreement = filter_agreement(anns_acc, agreement=3)

    ious = [ann["completed_task_sim"]["iou"] for ann in anns]
    ious_acc = [ann["completed_task_sim"]["iou"] for ann in anns_acc]
    ious_acc_with_agreement = [ann["completed_task_sim"]["iou"] for ann in anns_acc_with_agreement]

    name = "All"
    if cat_id:
        name = coco.cats[cat_id]["name"]
    print("{} SQ:".format(name), np.mean(ious_acc))

    title = "Accepted Annotations"
    if cat_id:
        title += " (category: {})".format(coco.cats[cat_id]["name"])
    xlabel = "IOU"
    ylabel = "Counts"
    h0,_,_ = plt.hist(ious, bins=10, stacked=True, range=(0,1), color="red")
    h1,_,_ = plt.hist(ious_acc, bins=10, stacked=True, range=(0,1), color="green")
    h2,_,_ = plt.hist(ious_acc_with_agreement, bins=10, stacked=True, range=(0,1), color=(0,1,0))
    
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.savefig('plots/{}.png'.format(title))
    # plt.show()
    plt.close()

def plot_time_histogram(coco):
    anns = coco.dataset["annotations"]
    times = [ann["completed_task"]["annotationTime"] for ann in anns]
    title = "Annotation Times"
    xlabel = "Times"
    ylabel = "Counts"
    h0,_,_ = plt.hist(times, bins="auto")

    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.savefig('plots/{}.png'.format(title))
    # plt.show()
    plt.close()


def filter_agreement(anns, agreement):
    counts = {}
    for ann in anns:
        key = str(ann["segmentation"])
        if key not in counts:
            counts[key] = 0
        counts[key] += 1

    anns = [ann for ann in anns if counts[str(ann["segmentation"])] >= agreement]
    return anns

def plot_scatter(x, y, title="Title", xlabel="x label", ylabel="y label"):
    colors = (0,0,0)
    area = np.pi*3
    plt.scatter(x, y, s=area, c=colors, alpha=0.5)
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.show()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-f', '--ann_fn', type=str)
    args = parser.parse_args()

    coco = COCO(args.ann_fn)

    plot_time_histogram(coco)
    plot_histogram(coco)
    for cat_id in coco.cats:
        plot_histogram(coco, cat_id)
