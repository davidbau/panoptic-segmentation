import os

from visualize import visualize_video
from video_annotation_old import VideoAnnotation, read_file


def main(vid_list, ann_dir, out_dir, show=False):
    VID_DIR = "./data/VIRAT/videos_original"

    for vid_name in vid_list:
        out_fn = os.path.join(out_dir, vid_name + ".mp4")
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
        visualize_video(vid_fn, vid_ann, out_fn=out_fn, show=show)

def run_old(out_dir, show=False):
    VID_LIST = "./data/VIRAT/docs/list_release2.0.txt"
    ANN_DIR = "./data/VIRAT/annotations"

    vid_list = read_file(VID_LIST)
    main(vid_list, ANN_DIR, out_dir, show=show)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('-o', '--output_dir', type=str, default="./data/videos/vis_old")
    parser.add_argument('-s', '--show', action='store_true')
    args = parser.parse_args()

    run_old(args.output_dir, show=args.show)