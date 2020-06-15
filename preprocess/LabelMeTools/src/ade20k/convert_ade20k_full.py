import os
import argparse
import numpy as np
import glob
import cv2

from scipy.io import loadmat

from coco_format import *

def make_ade20k_annotations(ann_dir, im_list):
    annotations = []
    for i, im_name in enumerate(im_list):
        print(i, im_name, len(annotations))

        ann_path = os.path.join(ann_dir, im_name).replace('.jpg', '_seg.png')

        # Parts not handled
        # parts_path = os.path.join(ann_dir, im_name).replace('.jpg', '_seg_parts_N.png')

        ann_image = cv2.imread(ann_path)
        if ann_image is None:
            print("Skipping", ann_path)
            continue
        
        ins_mask = ann_image[:,:,0] # B
        g_mask = ann_image[:,:,1] # G
        r_mask = ann_image[:,:,2] # R
        cat_mask = (r_mask.astype(int) / 10) * 256 + g_mask

        for ins in np.unique(ins_mask):
            if ins == 0:
                continue
            mask = (ins_mask == ins)
            cat = np.sum(cat_mask[mask]) / np.sum(mask)
            crowd = 0
            
            ann = make_ann(mask, iscrowd=crowd)
            ann["image_id"] = i + 1
            ann["category_id"] = int(cat)
            ann["id"] = len(annotations) + 1
            annotations.append(ann)
    return annotations

def make_im_list(filenames, folders, split):
    im_list = []
    for filename, folder in zip(filenames, folders):
        im_path = os.path.join(folder, filename)
        if "ADE_" + split in im_path:
            im_list.append(im_path)
    return im_list

def correct_annotation_sizes(coco):
    for imgId in tqdm(coco.imgs):
        img = coco.imgs[imgId]
        h = img["height"]
        w = img["width"]

        annIds = coco.getAnnIds(imgIds=[imgId])
        anns = coco.loadAnns(annIds)
        for ann in anns:
            segm = ann["segmentation"]
            if segm["size"] != [h,w]:
                mask = COCOmask.decode(segm)
                new_mask = cv2.resize(mask, (w, h), cv2.INTER_NEAREST)
                new_ann = make_ann(new_mask)

                ann["segmentation"] = new_ann["segmentation"]
                ann["area"] = new_ann["area"]
                ann["bbox"] = new_ann["bbox"]

def open_mat_file(ann_dir):
    mat_file = os.path.join(ann_dir, "index_ade20k.mat")
    mat_contents = loadmat(mat_file, squeeze_me=True)["index"]

    filename = mat_contents["filename"]
    folder = mat_contents["folder"]
    typeset = mat_contents["typeset"]
    objectIsPart = mat_contents["objectIsPart"]
    objectPresence = mat_contents["objectPresence"]
    objectcounts = mat_contents["objectcounts"]
    objectnames = mat_contents["objectnames"]
    proportionClassIsPart = mat_contents["proportionClassIsPart"]
    scene = mat_contents["scene"]
    wordnet_found = mat_contents["wordnet_found"]
    wordnet_level1 = mat_contents["wordnet_level1"]
    wordnet_synset = mat_contents["wordnet_synset"]
    wordnet_hypernym = mat_contents["wordnet_hypernym"]
    wordnet_gloss = mat_contents["wordnet_gloss"]
    wordnet_synonyms = mat_contents["wordnet_synonyms"]
    wordnet_frequency = mat_contents["wordnet_frequency"]


    contents = [filename, folder, objectnames]
    contents = [c.tolist().tolist() for c in contents]
    return contents

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-s', '--split', type=str, default="val")
    args = parser.parse_args()

    data_dir = "../data/ade20k"
    im_dir = os.path.join(data_dir, "images")
    ann_dir = os.path.join(data_dir, "annotations/full")

    # Open index mat file
    contents = open_mat_file(ann_dir)
    filenames = contents[0]
    folders = contents[1]

    # Make cat_list
    cat_list = contents[2]
    cat_list.insert(0, "__background__")

    # Make im_list
    im_list = make_im_list(filenames, folders, args.split)
    im_list_renamed = ["{}/{}".format(im_name.split("/")[2], os.path.basename(im_name)) for im_name in im_list]

    # Make annotations
    annotations = make_ade20k_annotations(ann_dir, im_list)
    images = make_images(im_list_renamed, im_dir)
    categories = make_categories(cat_list)
    out_fn = os.path.join(ann_dir, "../full_{}_wrong_size.json".format(args.split))
    save_ann_fn(images, annotations, categories, out_fn)
    print_ann_fn(out_fn)

    # Correct annotation sizes
    coco = COCO(out_fn)
    correct_annotation_sizes(coco)

    images = coco.dataset["images"]
    annotations = coco.dataset["annotations"]
    categories = coco.dataset["categories"]
    out_fn = os.path.join(ann_dir, "../full_{}.json".format(args.split))
    save_ann_fn(images, annotations, categories, out_fn)
    print_ann_fn(out_fn)

