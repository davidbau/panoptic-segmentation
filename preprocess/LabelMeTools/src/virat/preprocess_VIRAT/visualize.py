import argparse
import os
import cv2
import time
import numpy as np

from video_annotation import VideoAnnotation, read_file

_GRAY = (218, 227, 218)
_GREEN = (18, 127, 15)
_WHITE = (255, 255, 255)
_BLUE = (127, 18, 15)
_RED = (18, 15, 127)

def vis_mask(img, mask, col, alpha=0.4, show_border=True, border_thick=1):
    """Visualizes a single binary mask."""

    img = img.astype(np.float32)
    idx = np.nonzero(mask)

    img[idx[0], idx[1], :] *= 1.0 - alpha
    img[idx[0], idx[1], :] += alpha * col

    if show_border:
        _, contours, _ = cv2.findContours(
            mask.copy(), cv2.RETR_CCOMP, cv2.CHAIN_APPROX_NONE)
        cv2.drawContours(img, contours, -1, _WHITE, border_thick, cv2.LINE_AA)

    return img.astype(np.uint8)


def vis_class(img, pos, class_str, font_scale=0.35, color=_GREEN):
    """Visualizes the class."""
    img = img.astype(np.uint8)
    x0, y0 = int(pos[0]), int(pos[1])
    # Compute text size.
    txt = class_str
    font = cv2.FONT_HERSHEY_SIMPLEX
    ((txt_w, txt_h), _) = cv2.getTextSize(txt, font, font_scale, 1)
    # Place text background.
    back_tl = x0, y0 - int(1.3 * txt_h)
    back_br = x0 + txt_w, y0
    cv2.rectangle(img, back_tl, back_br, color, -1)
    # Show text.
    txt_tl = x0, y0 - int(0.3 * txt_h)
    cv2.putText(img, txt, txt_tl, font, font_scale, _GRAY, lineType=cv2.LINE_AA)
    return img


def vis_bbox(img, bbox, thick=1, color=_GREEN):
    """Visualizes a bounding box."""
    img = img.astype(np.uint8)
    (x0, y0, w, h) = bbox
    x1, y1 = int(x0 + w), int(y0 + h)
    x0, y0 = int(x0), int(y0)
    cv2.rectangle(img, (x0, y0), (x1, y1), color, thickness=thick)
    return img

def vis_image(im, names, bboxes, color=_GREEN):
    for name, bbox in zip(names, bboxes):
        im = vis_bbox(im, bbox, color=color)
        im = vis_class(im, (bbox[0], bbox[1] - 2), name, color=color)
    return im

def draw_objects(im, objs, color=_GREEN):
    names = []
    bboxes = []
    for obj in objs:
        name = obj["name"]
        if name != "null":
            names.append(name)
            bboxes.append(obj["bbox"])
    # print names
    im = vis_image(im, names, bboxes, color=color)
    return im

def draw_events(im, evts):
    names = []
    bboxes = []
    for evt in evts:
        name = evt["name"]
        if name != "null":
            names.append(name)
            bboxes.append(evt["bbox"])
    # print names
    im = vis_image(im, names, bboxes, color=_BLUE)
    return im

def visualize_video(vid_fn, vid_ann, out_fn='output.mp4', show=False):
    cap = cv2.VideoCapture(vid_fn)
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    length = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    print frame_height, frame_width, fps

    if not os.path.exists(os.path.dirname(out_fn)):
        os.makedirs(os.path.dirname(out_fn))
    # Define the codec and create VideoWriter object
    fourcc = cv2.VideoWriter_fourcc(*'MP4V')
    out = cv2.VideoWriter(out_fn, fourcc, fps, (frame_width, frame_height))

    frame_num = 0
    while(cap.isOpened()):
        ret, frame = cap.read()
        if not ret:
            break

        frame_num += 1
        if frame_num % 10 == 0:
            print "{}/{}".format(frame_num, length)
            objs = vid_ann.get_objects_at_frame(frame_num)
            evts = vid_ann.get_events_at_frame(frame_num)

            # Visualize objects
            frame = draw_objects(frame, objs)

            # Visualize events
            claimed = []
            for evt in evts:
                claimed.extend(evt["object_ids"])
            claimed_obj = [obj for obj in objs if obj["id"] in claimed]
            frame = draw_objects(frame, claimed_obj, color=_RED)
            frame = draw_events(frame, evts)

            out.write(frame)
            if show:
                cv2.imshow('frame',frame)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break

    cap.release()
    out.release()
    if show:
        cv2.destroyAllWindows()

def main(vid_list, ann_dir, out_dir, show=False):
    VID_DIR = "../data/virat/raw_data/VIRAT/videos_original"

    for vid_name in vid_list:
        vid_fn = os.path.join(VID_DIR, vid_name + ".mp4")
        ann_fn = os.path.join(ann_dir, vid_name + ".json")
        out_fn = os.path.join(out_dir, vid_name + ".mp4")
        if os.path.exists(out_fn):
            print "Already done", vid_name
            continue

        vid_ann = VideoAnnotation(ann_fn)
        if not vid_ann.loaded:
            print "Could not load annotations", ann_fn
            continue

        if not os.path.exists(vid_fn):
            print "Could not load video", vid_fn
            continue

        print vid_name
        vid_ann.expandEventBboxes() # Looks better
        visualize_video(vid_fn, vid_ann, out_fn=out_fn, show=show)

def run(indir, outdir, show=False):
    file_index = read_file(os.path.join(indir, "file-index.json"))
    vid_list = [os.path.splitext(k)[0] for k in file_index]
    vid_list.sort()
    main(vid_list, indir, outdir)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-i', '--indir', type=str, default="train")
    parser.add_argument('-o', '--outdir', type=str, default="../data/virat/")
    parser.add_argument('-s', '--show', action='store_true')
    args = parser.parse_args()

    if args.indir == "train":
        indir = "../data/virat/raw_data/VIRAT-V1_JSON_train-leaderboard_drop4_20180614"
    elif args.indir == "val":
        indir = "../data/virat/raw_data/VIRAT-V1_JSON_validate-leaderboard_drop4_20180614"

    run(indir, args.outdir, show=args.show)





