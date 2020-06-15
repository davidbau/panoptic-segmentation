import argparse
import os
import cv2
import numpy as np
from tqdm import tqdm

def get_im_list(im_dir):
    im_list = []
    for root, dirs, files in os.walk(im_dir):
        for name in files:
            if '.jpg' in name or '.png' in name:
                name = os.path.join(root, name)
                im_list.append(name)
    im_list.sort()
    return im_list

def make_video(im_list, out_fn):
    img = cv2.imread(im_list[0])
    frame_height = img.shape[0]
    frame_width = img.shape[1]
    fps = 30.0

    # Define the codec and create VideoWriter object
    fourcc = cv2.VideoWriter_fourcc(*'MP4V')
    out = cv2.VideoWriter(out_fn, fourcc, fps, (frame_width, frame_height))

    for im_name in tqdm(im_list):
        img = cv2.imread(im_name)
        out.write(img)

    out.release()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-i', '--input_dir', help='The input directory')
    parser.add_argument('-o', '--output_dir', help='The output directory', default="./output")
    args = parser.parse_args()

    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)

    im_list = get_im_list(args.input_dir)
    out_fn = os.path.join(args.output_dir, "video.mp4")
    make_video(im_list, out_fn)