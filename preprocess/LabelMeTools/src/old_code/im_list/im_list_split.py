import os
import argparse
import random
from tqdm import tqdm

def split_im_list(im_list, n):
    c = 0
    im_lists = []
    while c < len(im_list):
        im_lists.append(im_list[c:c+n])
        c += n
    return im_lists

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-m', '--max_num', type=int, default=None, help='Max number of images per im_list')
    parser.add_argument('-r', '--randomize')
    args = parser.parse_args()
    print(args)

    
    im_lists = [im_list]
    if args.max_num != None:
        im_lists = split_im_list(im_list, args.max_num)

    for i, im_list in enumerate(im_lists):
        out_fn = "images{}.txt".format(i)
        print("{} / {}".format(i+1, len(im_lists)), out_fn)
        write_im_list(im_list, out_fn)