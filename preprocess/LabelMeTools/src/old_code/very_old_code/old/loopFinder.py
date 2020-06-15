import os
import cv2
from scipy import misc
import numpy as np

from collections import deque

# Deprecated. Nothing uses this file

class LoopFinder:

    def __init__(self, export_path):
        self.export_path = export_path

    def processImage(self, image):
        loops = []
        start = (0,0)
        while True:
            start = self.findStart(image, start)
            if start == None:
                break
            path = self.bfsFindLoop(image, start)
            if path != None:
                loops.append(path)
                self.erase(path, image)
        return loops

    def visualizeLoops(self, loops, shape):
        loop_image = np.zeros((shape[0], shape[1], 3))
        for loop in loops:
            color = np.random.randint(0, 255, size=3)
            for node in loop:
                i,j = node
                loop_image[i][j] = color
        return loop_image

    def export(self, name, loops, shape):
        print "Exporting loop image:", self.export_path
        loop_image = self.visualizeLoops(loops, shape)
        fileName = "{}_loops.png".format(name)
        path = os.path.join(self.export_path,fileName)
        misc.imsave(path, loop_image)

    def erase(self, pixels, image):
        for pixel in pixels:
            i,j = pixel
            image[i][j] = 0

    def findStart(self,image, lastStart):
        h,w = image.shape
        for i in xrange(lastStart[0], h):
            for j in xrange(w):
                if image[i][j] > 0:
                    return (i,j)
        return None

    def bfsFindLoop(self, image, start):
        queue = deque()
        parent = {}

        queue.append(start)
        parent[start] = None
        u = None
        while len(queue) > 0:
            u =  queue.popleft()
            for v in self.children(u, parent[u], image):
                if v in parent:
                    # Found loop
                    path1 = self.backtrack(u, start, parent)
                    path2 = self.backtrack(v, start, parent)
                    path1.reverse()
                    path2.reverse()
                    # Find common ancestor and build loop
                    for i in xrange(len(path1)):
                        if path1[i] != path2[i]:
                            part1 = path1[i:]
                            part2 = path2[i-1:]
                            part2.reverse()
                            loop = part1 + part2
                            if len(loop) > 40:
                                # Only keep loops larger than 40 pixels
                                return loop
                            else:
                                break
                else:
                    parent[v] = u
                    queue.append(v)
        self.erase(parent.keys(), image)
        return None

    def backtrack(self, end, start, parent):
        path = [end]
        while end != start:
            end = parent[end]
            path.append(end)
        return path

    def children(self, node, parent_node, image):
        children = []
        h,w = image.shape
        i,j = node
        sides = [(i+1,j+1),(i,j+1),(i-1,j+1),(i+1,j),(i-1,j),(i+1,j-1),(i,j-1),(i-1,j-1)]
        neighbors = [(i+1,j+1),(i,j+1),(i-1,j+1),(i+1,j),(i-1,j),(i+1,j-1),(i,j-1),(i-1,j-1)]
        for neighbor in neighbors:
            i,j = neighbor
            if 0 <= i and i < h and 0 <= j and j < w:
                if image[i][j] > 0 and parent_node != (i,j):
                    children.append((i,j))
        return children

if __name__=="__main__":

    num = "3"
    image_num = num.zfill(8)
    image_name = "ADE_train_{}_boundary_0".format(image_num)
    image_path = "images/{}.png".format(image_name)
    image = misc.imread(image_path)

    # cv2.imshow("", image)
    # cv2.waitKey(0)

    export_path = "images/"
    converter = LoopFinder(export_path)
    loops = converter.processImage(image)
    converter.export(image_name, loops, image.shape)

