import os
import cv2
import time
import numpy as np

from video_annotation import VideoAnnotation, read_file

CROP_SIZE = 128
CONTEXT = 1

def stitch_images(images, shape=(3,3)):
    n = shape[0] * shape[1]
    h, w = (128, 128)
    if len(images) > 0:
        h, w = images[0].shape[:2]
    stitched = np.zeros((h*shape[0], w*shape[1], 3), dtype="uint8")
    for i, image in enumerate(images[:n]):
        x = i % shape[1]
        y = int(i / shape[1])
        stitched[y*h:(y+1)*h, x*w:(x+1)*w] = image
    return stitched

def crop_image(im, bbox):
    (x, y, w, h) = bbox
    x_c = x + w/2
    y_c = y + h/2
    m = max(h,w)
    m = m * CONTEXT
    m = int(m / 2) * 2 # Needs to be even
    x0 = min(max(0, x_c - m/2), im.shape[1])
    x1 = min(max(0, x_c + m/2), im.shape[1])
    y0 = min(max(0, y_c - m/2), im.shape[0])
    y1 = min(max(0, y_c + m/2), im.shape[0])
    crop = im[y0:y1, x0:x1]

    # Pad with zeros
    pad_l = -min(x_c - m/2, 0)
    pad_t = -min(y_c - m/2, 0)
    pad = np.zeros((m, m, 3), dtype='uint8')
    pad[pad_t:pad_t + crop.shape[0], pad_l:pad_l + crop.shape[1]] = crop

    resized = cv2.resize(pad, dsize=(CROP_SIZE, CROP_SIZE))
    return resized

def crop_objects(im, objs):
    names = []
    bboxes = []
    for obj in objs:
        name = obj["name"]
        if name != "null":
            name = "{}-{}".format(name, obj["id"])
            names.append(name)
            bboxes.append(obj["bbox"])
    out = {}
    for name, bbox in zip(names, bboxes):
        out[name] = crop_image(im, bbox)
    return out

def crop_events(im, evts):
    names = []
    bboxes = []
    for evt in evts:
        name = evt["name"]
        if name != "null":
            name = "{}-{}".format(name, evt["id"])
            names.append(name)
            bboxes.append(evt["bbox"])
    out = {}
    for name, bbox in zip(names, bboxes):
        out[name] = crop_image(im, bbox)
    return out

def crop_video(vid_fn, vid_ann, out_dir):
    cap = cv2.VideoCapture(vid_fn)
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    length = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    print frame_height, frame_width, fps

    # Define the codec and create VideoWriter object
    fourcc = cv2.VideoWriter_fourcc(*'MP4V')
    out_writers = {}

    frame_num = 0
    while(cap.isOpened()):
        ret, frame = cap.read()
        if not ret:
            break

        frame_num += 1
        if frame_num % 1 == 0:
            print frame_num
            objs = vid_ann.get_objects_at_frame(frame_num)
            evts = vid_ann.get_events_at_frame(frame_num)

            # Get everything cropped
            cropped_objs = crop_objects(frame, objs)
            cropped_evts = crop_events(frame, evts)
            cropped_objs_in_evts = {}
            for evt in evts:
                objs_in_evt = [obj for obj in objs if obj["id"] in evt["object_ids"]]
                cropped = crop_objects(frame, objs_in_evt)
                for name in cropped:
                    new_name = "{}-{}".format(evt["name"], name)
                    cropped_objs_in_evts[new_name] = cropped[name]

            # Prepare to write to files
            out_frames = {}
            for name in cropped_objs:
                out_fn = os.path.join(out_dir, name + ".mp4")
                out_frames[out_fn] = cropped_objs[name]
            for name in cropped_evts:
                out_fn = os.path.join(out_dir, name + ".mp4")
                out_frames[out_fn] = cropped_evts[name]
            for name in cropped_objs_in_evts:
                out_fn = os.path.join(out_dir, name + ".mp4")
                out_frames[out_fn] = cropped_objs_in_evts[name]

            # Write frames to their respective files
            for out_fn in out_frames:
                if out_fn not in out_writers:
                    print out_fn
                    if not os.path.exists(os.path.dirname(out_fn)):
                        os.makedirs(os.path.dirname(out_fn))
                    out_writers[out_fn] = cv2.VideoWriter(out_fn, fourcc, fps, (CROP_SIZE, CROP_SIZE))
                frame = out_frames[out_fn]
                out_writers[out_fn].write(frame)

    cap.release()
    for out_fn in out_writers:
        out_writers[out_fn].release()

def main(vid_list, ann_dir, out_dir):
    VID_DIR = "./data/VIRAT/videos_original"

    for vid_name in vid_list:
        out_fn = os.path.join(out_dir, vid_name)
        if os.path.exists(out_fn):
            print "Already done", vid_name
            continue

        vid_ann = VideoAnnotation(vid_name, ann_dir)
        if not vid_ann.loaded:
            print "Could not load annotations", vid_name
            continue

        vid_fn = os.path.join(VID_DIR, vid_name + ".mp4")
        if not os.path.exists(vid_fn):
            print "Could not load video", vid_name
            continue

        print vid_name
        crop_video(vid_fn, vid_ann, out_dir=out_fn)

def get_vid_list(ann_dir):
    dic = read_file(os.path.join(ann_dir, "file-index.json"))
    vid_list = [os.path.splitext(k)[0] for k in dic]
    vid_list.sort()
    return vid_list


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('-a', '--ann_dir', type=str, default="val")
    parser.add_argument('-o', '--out_dir', type=str, default="./data/videos/cropped/")
    parser.add_argument('-s', '--crop_size', type=int, default=128)
    parser.add_argument('-c', '--context', type=float, default=1.)
    args = parser.parse_args()

    CROP_SIZE = args.crop_size
    CONTEXT = args.context

    if args.ann_dir == "train":
        args.ann_dir = "./data/VIRAT-V1_JSON_train-leaderboard_drop4_20180614"
    elif args.ann_dir == "val":
        args.ann_dir = "./data/VIRAT-V1_JSON_validate-leaderboard_drop4_20180614"

    vid_list = get_vid_list(args.ann_dir)
    main(vid_list, args.ann_dir, args.out_dir)
