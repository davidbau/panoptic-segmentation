import argparse
import os
import cv2
from pycocotools.coco import COCO

from coco_format import *
from virat_format import VideoAnnotation, read_file

def process_video(vid_fn, img_dir):
    cap = cv2.VideoCapture(vid_fn)
    vid_name = os.path.splitext(os.path.basename(vid_fn))[0]
    length = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    frame_num = 0
    while(cap.isOpened()):
        ret, frame = cap.read()
        if not ret:
            break
        if (frame_num % 3 != 0):
            frame_num += 1
            continue

        frame_name = "{}/{}/{}_{}.jpg".format(args.split, vid_name, vid_name, str(frame_num).zfill(6))
        print("{}/{}".format(frame_num, length), frame_name)

        img_path = os.path.join(img_dir, frame_name)
        if not os.path.exists(os.path.dirname(img_path)):
            os.makedirs(os.path.dirname(img_path))
        cv2.imwrite(img_path, frame)
        frame_num += 1
    cap.release()

def process_annotations(vid_fn, ann_fn, ann_dir):
    cap = cv2.VideoCapture(vid_fn)
    vid_name = os.path.splitext(os.path.basename(vid_fn))[0]
    length = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    cap.release()

    vid_ann = VideoAnnotation(ann_fn)
    im_list = []
    annotations = []
    cat_list = ["__background__"]

    img_id = 0
    for frame_num in range(0, length, 3):
        objs = vid_ann.get_objects_at_frame(frame_num)
        # evts = vid_ann.get_events_at_frame(frame_num)
        # objs = objs + evts

        for obj in objs:
            name = obj["name"]
            bbox = obj["bbox"]
            if name not in cat_list:
                cat_list.append(name)
            cat_id = cat_list.index(name)

            ann = {}
            ann["image_id"] = img_id
            ann["id"] = len(annotations)
            ann["bbox"] = bbox
            ann["category_id"] = cat_id
            ann["iscrowd"] = 0
            ann["area"] = bbox[2] * bbox[3]
            ann["num_keypoints"] = 0
            ann["keypoints"] = [0]*17*3
            annotations.append(ann)

        frame_name = "{}/{}/{}_{}.jpg".format(args.split, vid_name, vid_name, str(frame_num).zfill(6))
        im_list.append(frame_name)
        img_id += 1

    images = make_images(im_list, shape=(frame_height, frame_width))
    categories = make_categories(cat_list)

    out_fn = os.path.join(ann_dir, args.split, vid_name + ".json")
    save_ann_fn(images, annotations, categories, out_fn)
    open_coco(out_fn)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-s', '--split', type=str, default="train")
    parser.add_argument('-o', '--outdir', type=str, default="../data/virat/")
    args = parser.parse_args()

    VID_DIR = "../data/virat/raw_data/VIRAT/videos_original"
    if args.split == "train":
        indir = "../data/virat/raw_data/VIRAT-V1_JSON_train-leaderboard_drop4_20180614"
    elif args.split == "val":
        indir = "../data/virat/raw_data/VIRAT-V1_JSON_validate-leaderboard_drop4_20180614"

    file_index = read_file(os.path.join(indir, "file-index.json"))
    vid_list = [os.path.splitext(k)[0] for k in file_index]
    vid_list.sort()

    img_dir = os.path.join(args.outdir, "images")
    ann_dir = os.path.join(args.outdir, "annotations")
    for vid_name in vid_list:
        print(vid_name)
        vid_fn = os.path.join(VID_DIR, vid_name + ".mp4")
        ann_fn = os.path.join(indir, vid_name + ".json")

        if not os.path.exists(vid_fn):
            print("Could not load video", vid_fn)
            continue

        if not os.path.exists(ann_fn):
            print("Could not load annotations", ann_fn)
            continue

        process_video(vid_fn, img_dir)
        process_annotations(vid_fn, ann_fn, ann_dir)