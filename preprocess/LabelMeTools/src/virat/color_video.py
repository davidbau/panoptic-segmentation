import argparse
import os
import cv2
import numpy as np
import colorsys
from tqdm import tqdm

def get_frames(video_path):
    frames = []

    # Load input video
    cap = cv2.VideoCapture(video_path)
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    length = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    print(fps)

    for i in tqdm(range(length)):
        ret, img = cap.read()
        if img is None:
            continue
        frames.append(img)
    cap.release()
    return frames

def save_frames(frames, out_fn):
    frame_height = frames[0].shape[0]
    frame_width = frames[0].shape[1]
    fps = 30.0

    # Define the codec and create VideoWriter object
    fourcc = cv2.VideoWriter_fourcc(*'MP4V')
    out = cv2.VideoWriter(out_fn, fourcc, fps, (frame_width, frame_height))

    for frame in tqdm(frames):
        out.write(frame)
    out.release()

def add_color_to_depth_video(depth_video, segm_video, out_fn):
    depth_frames = get_frames(depth_video)
    segm_frames = get_frames(segm_video)
    vis_frames = []
    for depth, segm in tqdm(zip(depth_frames, segm_frames)):
        depth = cv2.cvtColor(depth, cv2.COLOR_BGR2GRAY)
        segm = cv2.cvtColor(segm, cv2.COLOR_BGR2GRAY)

        mask = segm == 1 # sky
        depth[mask] = 0
        depth = np.sqrt(depth / np.amax(depth))
        depth = np.array(depth * 255, dtype='uint8')

        img_color = cv2.applyColorMap(depth, cv2.COLORMAP_JET)
        img_color[mask] = 0

        vis_frames.append(img_color)
    save_frames(vis_frames, out_fn)

def add_color_to_segm_video(orig_video, segm_video, out_fn):
    orig_frames = get_frames(orig_video)
    segm_frames = get_frames(segm_video)
    vis_frames = []
    for orig, segm in tqdm(zip(orig_frames, segm_frames)):
        segm = cv2.cvtColor(segm, cv2.COLOR_BGR2GRAY)
        segm = np.array(segm * 6, dtype='uint8')
        segm_color = cv2.applyColorMap(segm, cv2.COLORMAP_HSV)
        img_color = cv2.addWeighted(orig, 0.7, segm_color, 0.3, 0) 

        vis_frames.append(img_color)
    save_frames(vis_frames, out_fn)

if __name__ == '__main__':
    # orig_video = "videos/highway.mp4"
    # segm_video = "videos/highway_segm.mp4"
    # depth_video = "videos/highway_depth.mp4"
    # depth_color_video = "videos/highway_depth_color.mp4"
    # segm_color_video = "videos/highway_segm_color.mp4"

    orig_video = "videos/dashcam.mp4"
    segm_video = "videos/dashcam_segm.mp4"
    depth_video = "videos/dashcam_depth.mp4"
    depth_color_video = "videos/dashcam_depth_color.mp4"
    segm_color_video = "videos/dashcam_segm_color.mp4"

    # add_color_to_depth_video(depth_video, segm_video, depth_color_video)
    add_color_to_segm_video(orig_video, segm_video, segm_color_video)



