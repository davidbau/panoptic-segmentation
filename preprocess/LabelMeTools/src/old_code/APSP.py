import heapq
import cv2
import numpy as np

class PathFinder():

    def __init__(self, top, goal, resolution=1):
        self.resolution = resolution
        self.top = cv2.resize(top, (0,0), fx=resolution, fy=resolution)
        self.goal = self.toRes(goal)

    def toRes(self, points):
        ps = np.array(points) * self.resolution
        ps[:,0] = np.clip(ps[:,0], 0, self.top.shape[1]-1)
        ps[:,1] = np.clip(ps[:,1], 0, self.top.shape[0]-1)
        ps = np.array(ps, dtype='int')
        return ps
    def toOri(self, points):
        scale = 1 / self.resolution
        ps = np.array(points) * scale + scale/2
        ps = np.array(ps, dtype='int')
        return ps 

    def findPath(self, start, end):
        pass

    def findPathToGoal(self, start):
        p = self.toRes([start])[0]
        path = [p]
        while True:
            parent = self.parents[p[1], p[0]]
            x = parent[0]
            y = parent[1]
            if x == 0 and y == 0:
                break
            path.append((x,y))
            p = (x,y)
        return self.toOri(path)

    def allPointShortestPath(self):
        h,w = self.top.shape
        self.distances = np.full((h,w), np.inf)
        self.parents = np.zeros((h, w, 2), dtype='int')

        queue = []
        for p in self.goal:
            self.distances[p[1], p[0]] = 0
            heapq.heappush(queue, (0, p))

        c = 0
        while len(queue) != 0:
            p_dist, p = heapq.heappop(queue)
            dists = getDistToNeighbors(self.top, p)
            for n in dists:
                n_dist = dists[n] + p_dist
                if n_dist < self.distances[n[1], n[0]]:
                    self.distances[n[1], n[0]] = n_dist
                    self.parents[n[1], n[0]] = p
                    heapq.heappush(queue, (n_dist, n))

            c += 1
            if c % 1000 == 0:
                print 1.*c/(h*w)


def drawPath(img, path):
    if np.ndim(img) == 2:
        img = np.stack([img,img,img], axis=-1)
    color = (0,0,255)
    for p in path:
        cv2.circle(img, tuple(p), 1, color, -1)
    return img

def getDistToNeighbors(top, p):
    h,w = top.shape
    neighborDists = {}
    dirs = [(0,1), (0,-1), (1,0), (-1,0), (1,1), (1,-1), (-1,1), (-1,-1)]
    for d in dirs:
        x = d[0] + p[0]
        y = d[1] + p[1]
        if x < 0 or x >=w or y < 0 or y >= h:
            continue
        neighborDists[(x,y)] = top[y,x] * np.linalg.norm(d)
    return neighborDists

def getTopography(img):
    top = cv2.blur(img,(5,5))
    top = cv2.Laplacian(top, cv2.CV_64F)
    top = np.clip(top, 0, 255)
    top /= np.max(top)
    top = 1 - top
    top = np.square(top)
    top = np.square(top)
    return top

def show(img):
    print np.min(img), np.max(img)
    img = img.copy()
    cv2.imshow("img", img)
    cv2.waitKey(0)



if __name__ == "__main__":
    img = cv2.imread("./data/ade20k/images/validation/ADE_val_00000002.jpg", 0)
    # img = img[:200,:200]
    h,w = img.shape

    top = getTopography(img)
    # canny = cv2.Canny(img,100,200)
    # top = np.array(1 - canny/np.max(canny), dtype="float")
    show(top)

    goal = [(w/2,h/2)]
    pathFinder = PathFinder(top, goal, resolution=0.5)
    pathFinder.allPointShortestPath()

    xs = np.random.randint(w, size=100)
    ys = np.random.randint(h, size=100)
    vis = cv2.resize(pathFinder.top, (w,h))
    for x,y in zip(xs,ys):
        path = pathFinder.findPathToGoal((x,y))
        vis = drawPath(vis, path)
    show(vis)
