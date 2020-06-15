import os
import random
import cv2
import glob
import numpy as np
from shutil import copyfile

def sort_videos(inp_dir, out_dir):
    vid_dirs = glob.glob(inp_dir + "*/")
    for vid_dir in vid_dirs:
        print vid_dir
        vid_name = os.path.basename(os.path.normpath(vid_dir))
        clips = [x for x in os.listdir(vid_dir) if ".mp4" in x]
        for clip in clips:
            split = clip.split('-')
            name = '-'.join(split[:-1])
            rest = split[-1]

            src = os.path.join(vid_dir, clip)
            dest = os.path.join(name, "{}-{}".format(vid_name, rest))
            dest = os.path.join(out_dir, dest)
            if not os.path.exists(os.path.dirname(dest)):
                os.makedirs(os.path.dirname(dest))

            copyfile(src, dest)

def stitch_frames(frames, shape=(4,5)):
    CROP_SIZE = None
    for frame in frames:
        if frame is not None:
            CROP_SIZE = frame.shape[0]
            break
    if CROP_SIZE is None:
        return None

    h,w = shape
    out = np.zeros((CROP_SIZE*h, CROP_SIZE*w, 3), dtype='uint8')
    for n, frame in enumerate(frames):
        if frame is None:
            continue
        y = (n / w) * CROP_SIZE
        x = (n % w) * CROP_SIZE
        out[y:y+CROP_SIZE, x:x+CROP_SIZE] = frame
    return out

def visualize_clips(clips, out_fn, shape=(4,5), show=False):
    n = shape[0] * shape[1]
    readers = [None] * n
    out = None

    for i in range(n):
        if len(clips) != 0:
            readers[i] = cv2.VideoCapture(clips.pop())

    for _ in range(2000):
        frames = []
        while len(frames) < n:
            i = len(frames)
            if readers[i] is None:
                frames.append(None)
            else:
                ret, frame = readers[i].read()
                if ret:
                    frames.append(frame)
                else:
                    readers[i].release()
                    if len(clips) != 0:
                        readers[i] = cv2.VideoCapture(clips.pop())
                    else:
                        readers[i] = None

        out_frame = stitch_frames(frames, shape=shape)
        if out_frame is None:
            break

        if out is None:
            # Define the codec and create VideoWriter object
            fourcc = cv2.VideoWriter_fourcc(*'MP4V')
            out = cv2.VideoWriter(out_fn, fourcc, 30, (out_frame.shape[1], out_frame.shape[0]))
        out.write(out_frame)

        if show:
            cv2.imshow('frame',out_frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    out.release()
    for reader in readers:
        if reader is not None:
            reader.release()

def vis_videos(sorted_dir, show=False):
    vid_dirs = glob.glob(sorted_dir + "*/")
    for vid_dir in vid_dirs:
        print vid_dir
        name = os.path.basename(os.path.normpath(vid_dir))
        clip_paths = glob.glob(os.path.join(vid_dir, "*.mp4"))
        random.shuffle(clip_paths)

        out_fn = os.path.join(sorted_dir, name + ".mp4")
        visualize_clips(clip_paths, out_fn, show=show)

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('-i', '--input_dir', type=str, default="./data/videos/cropped/")
    parser.add_argument('-o', '--output_dir', type=str, default="./data/videos/cropped_sorted/")
    parser.add_argument('-s', '--show', action='store_true')
    args = parser.parse_args()

    print "Sorting..."
    sort_videos(args.input_dir, args.output_dir)

    print "Visualizing..."
    vis_videos(args.output_dir, show=args.show)