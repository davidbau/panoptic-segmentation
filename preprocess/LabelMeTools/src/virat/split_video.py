import argparse
import os
import cv2
import numpy as np
from tqdm import tqdm

def split_video(video_path, output_dir):
    # Load input video
    cap = cv2.VideoCapture(video_path)
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    length = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    for i in tqdm(range(length)):
        ret, img = cap.read()
        if img is None:
            continue

        out_fn = os.path.join(output_dir, "{:06d}.jpg".format(i))
        cv2.imwrite(out_fn, img)

    cap.release()

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-v', '--video_path', help='The video path')
    parser.add_argument('-o', '--output_dir', help='The output dir', default="./output")
    args = parser.parse_args()

    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)

    split_video(args.video_path, args.output_dir)