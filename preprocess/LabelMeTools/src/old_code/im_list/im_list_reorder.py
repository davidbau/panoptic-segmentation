import os
import argparse

from im_list_make import *

def read_im_list(path):
    print(path)
    with open(path) as f:
        im_list = f.read().splitlines()
        return im_list


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-d', '--im_list_dir', type=str, default="../data/places/im_lists_old", help='Root im_lists directory')
    parser.add_argument('-m', '--max_num', type=int, default=None, help='Max number of images per im_list')
    args = parser.parse_args()

    im_list = []
    fns = os.listdir(args.im_list_dir)
    for i in range(len(fns)):
        fn = "images{}.txt".format(i)
        path = os.path.join(args.im_list_dir, fn)
        im_list += read_im_list(path)

    print(len(im_list))

    im_lists = split_im_list(im_list, args.max_num)
    for i, im_list in enumerate(im_lists):
        out_fn = "images%03d.txt" % i
        print("{} / {}".format(i+1, len(im_lists)), out_fn)
        write_im_list(im_list, out_fn)
