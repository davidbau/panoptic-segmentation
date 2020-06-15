import os
import json
from scipy import misc

from maskToPolygons import MaskToPolygons

path = "/Users/hujh/Documents/UROP_Torralba/LabelMe/datasets/examples/annotations/original_annotations"
polygon_folder = "{}/../polygons".format(path)
temp_folder = "{}/temp".format(path)
converter = MaskToPolygons()

for i in xrange(200):
    num = str(i)
    image_num = num.zfill(8)
    image_name = "ADE_train_{}".format(image_num)
    image_path = "{}/{}.png".format(path,image_name)
    if os.path.exists(image_path):
        print "Processing image", num
        image = misc.imread(image_path)
        categoryToPolygons, debug = converter.processMask(image)

        data = {}
        for category in categoryToPolygons:
            counter = 0
            for polygon in categoryToPolygons[category]:
                key = "{}#{}".format(category, counter)
                # Flip (h,w) to (x,y)
                flipped = [[point[1], point[0]] for point in polygon]
                data[key] = polygon
                counter += 1

        polygon_file = "/{}/{}-polygons.json".format(polygon_folder, image_name)

        with open(polygon_file, 'w') as outfile:
            json.dump(data, outfile)

