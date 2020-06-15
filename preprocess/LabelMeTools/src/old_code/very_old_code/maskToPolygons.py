import sys
import cv2
import matplotlib.pyplot as plt
import numpy as np

class MaskToPolygons:

    def __init__(self):
        self.min_size = 100

    def process(self, category_mask):
        categoryToMask = self.mapCategoryToMask(category_mask)
        categoryToPolygons = {}

        debug = np.zeros(category_mask.shape)
        # debug = None

        for cat in categoryToMask.keys():
            mask = categoryToMask[cat]
            mask = mask.astype(np.uint8)

            contours, hierarchy = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_TC89_L1)

            if contours:
                contours = [c for c,h in zip(contours, hierarchy[0]) if cv2.contourArea(c) > self.min_size and h[3] < 0]
                
                # Approximate
                approxs = []
                for cnt in contours:
                    epsilon = 0.001*cv2.arcLength(cnt,True)
                    approx = cv2.approxPolyDP(cnt,epsilon,True)
                    approxs.append(approx)
                contours = approxs

                # Visualize
                for c in contours:
                    color = np.random.randint(0, 255, size=3)
                    cv2.drawContours(debug, [c], -1, (color[0], color[1], color[2]), 3)

                polygons = []
                for c in contours:
                    n = c.shape[0]
                    c = c.reshape((n,2))
                    polygon = c.tolist()
                    polygons.append(polygon)

                categoryToPolygons[cat] = polygons

        if 0 in categoryToPolygons:
            del categoryToPolygons[0]
        return categoryToPolygons, debug

    def mapCategoryToMask(self, image):
        categoryToMask = {}
        for i in xrange(image.shape[0]):
            for j in xrange(image.shape[1]):
                c = image[i][j]
                if c not in categoryToMask:
                    new_mask = np.zeros(image.shape)
                    categoryToMask[c] = new_mask
                categoryToMask[c][i][j] = 1
        return categoryToMask

if __name__=="__main__":
    name = sys.argv[1]
    category_mask = cv2.imread(name, 0)

    converter = MaskToPolygons()
    categoryToPolygons, debug = converter.process(category_mask)
    plt.imshow(debug)
    plt.show()

