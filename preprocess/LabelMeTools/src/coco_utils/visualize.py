import os
import cv2
import random
import argparse
import numpy as np
from tqdm import tqdm

from pycocotools.coco import COCO
from pycocotools import mask as COCOmask

_GRAY = (218, 227, 218)
_GREEN = (18, 127, 15)
_WHITE = (255, 255, 255)
_BLUE = (127, 18, 15)
_RED = (18, 15, 127)
COLOR_MAP = {}

def get_color(name):
    if name not in COLOR_MAP:
        r =  random.randint(0, 255)
        g =  random.randint(0, 255)
        b =  random.randint(0, 255)
        COLOR_MAP[name] = (b,g,r)
    return COLOR_MAP[name]

def vis_mask(img, mask, alpha=0.4, show_border=True, border_thick=1, color=_GREEN):
    """Visualizes a single binary mask."""
    img = img.astype(np.float32)
    idx = np.nonzero(mask)

    img[idx[0], idx[1], :] *= 1.0 - alpha
    img[idx[0], idx[1], :] += alpha * np.array(color)

    if show_border:
        contours, hierarchy = cv2.findContours(
            mask.copy(), cv2.RETR_CCOMP, cv2.CHAIN_APPROX_NONE)[-2:]
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

def vis_keypoints(img, keypoints, skeleton, radius=4, thick=2, color=_WHITE):
    """Visualizes keypoints."""
    img = img.astype(np.uint8)

    keypoints = np.array(keypoints).reshape(-1, 3)
    visible_kps = {}

    # Draw keypoints
    for i, kp in enumerate(keypoints):
        x,y,v = kp
        if v != 0:
            cv2.circle(img, (int(x), int(y)), radius, color, thickness=-1)
            visible_kps[i+1] = (x,y)
    # Draw skeleton
    for l in skeleton:
        if l[0] in visible_kps and l[1] in visible_kps:
            cv2.line(img, visible_kps[l[0]], visible_kps[l[1]], color, thickness=thick)
    return img

def vis_ann(coco, image, ann, showClass=True, showBbox=True, showSegm=True, showKps=True, crop=False):
    cat = coco.cats[ann["category_id"]]
    name = cat["name"]
    bbox = [0,0,0,0]
    mask = None
    keypoints = None
    skeleton = None

    if "bbox" in ann:
        bbox = ann["bbox"]
    elif "segmentation" in ann:
        bbox = COCOmask.toBbox(ann["segmentation"])
    if "score" in ann:
        name += " %.2f" % ann["score"]
    if "segmentation" in ann:
        mask = coco.annToMask(ann)
    if "keypoints" in ann:
        keypoints = ann["keypoints"]
        skeleton = cat["skeleton"]

    # Visualize
    color = get_color(cat["name"])
    if showClass:
        image = vis_class(image, (bbox[0], bbox[1] - 2), name, color=color)
    if showBbox:
        image = vis_bbox(image, bbox, color=color)
    if showSegm and mask is not None:
        image = vis_mask(image, mask, color=color)
    if showKps and keypoints is not None and skeleton is not None:
        image = vis_keypoints(image, keypoints, skeleton)
    if crop:
        image = crop_square(image, bbox)
    return image

def load_image(coco, im_dir, img):
    img_fn = os.path.join(im_dir, img["file_name"])
    image = cv2.imread(img_fn)
    if image is None:
        print("Warning: Could not find ", img_fn)
        image = np.zeros((img["height"], img["width"], 3))
    return image

def crop_square(image, bbox, margin=2, crop_size=256):
    x, y, w, h = bbox
    if min(h,w) <= 1:
        # Bad bbox
        return np.zeros((crop_size, crop_size, 3), dtype='uint8')

    x_c = x + w/2
    y_c = y + h/2
    m = max(h,w) * margin
    m = min(m, max(image.shape[0], image.shape[1]))
    m = int(m / 2) * 2 # Needs to be even
    x0 = int(min(max(0, x_c - m/2), image.shape[1]))
    x1 = int(min(max(0, x_c + m/2), image.shape[1]))
    y0 = int(min(max(0, y_c - m/2), image.shape[0]))
    y1 = int(min(max(0, y_c + m/2), image.shape[0]))
    crop = image[y0:y1, x0:x1]

    # Pad with zeros
    pad_l = int(-min(x_c - m/2, 0))
    pad_t = int(-min(y_c - m/2, 0))
    pad = np.zeros((m, m, 3), dtype='uint8')
    pad[pad_t:pad_t + crop.shape[0], pad_l:pad_l + crop.shape[1]] = crop
    resized = cv2.resize(pad, dsize=(crop_size, crop_size))
    return resized

def vis_coco(coco, im_dir, out_dir):
    html_dir = os.path.join(out_dir, "html")
    if not os.path.exists(html_dir):
        os.makedirs(html_dir)
    print("Visualizing to", html_dir)

    # Clear html directory
    for html_fn in os.listdir(html_dir):
        os.remove(os.path.join(html_dir, html_fn))

    for img in tqdm(coco.dataset["images"]):
        annIds = coco.getAnnIds(imgIds=[img["id"]])
        anns = coco.loadAnns(annIds)

        # Image visualization
        image = load_image(coco, im_dir, img)
        for ann in anns:
            image = vis_ann(coco, image, ann)

        out_fn = os.path.join(out_dir, img["file_name"])
        if not os.path.exists(os.path.dirname(out_fn)):
            os.makedirs(os.path.dirname(out_fn))
        cv2.imwrite(out_fn, image)

        html_fn = os.path.join(html_dir, "all_images.html")
        add_to_html(html_fn, out_fn)

        # Instance visualization
        for ann in anns:
            cat_name = coco.cats[ann["category_id"]]["name"]
            instance = load_image(coco, im_dir, img)
            instance = vis_ann(coco, instance, ann, showClass=False, showBbox=False, showSegm=True, showKps=False, crop=True)

            out_fn = os.path.join(out_dir, "instances", "{}/{}.jpg".format(cat_name, ann["id"]))
            if not os.path.exists(os.path.dirname(out_fn)):
                os.makedirs(os.path.dirname(out_fn))
            cv2.imwrite(out_fn, instance)

            html_fn = os.path.join(html_dir, "{}.html".format(cat_name))
            add_to_html(html_fn, out_fn)

def add_to_html(html_fn, out_fn):
    if not os.path.exists(os.path.dirname(html_fn)):
        os.makedirs(os.path.dirname(html_fn))

    path = os.path.relpath(out_fn, os.path.dirname(html_fn))
    with open(html_fn, "a") as f:
        tag = "<img src=\"" + path +"\" height=\"300\">"
        f.write(tag + "\n")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-f', '--ann_fn', type=str, help='Annotation file')
    parser.add_argument('-o', '--out_dir', type=str, default=None, help='Output visualization directory')
    parser.add_argument('-d', '--im_dir', type=str, default="/data/vision/torralba/ade20k-places/data", help='Images directory')
    args = parser.parse_args()
    if args.out_dir == None:
        args.out_dir = args.ann_fn.replace(".json", "_visualized")
    print(args)

    coco = COCO(args.ann_fn)
    vis_coco(coco, args.im_dir, args.out_dir)

