import argparse
import os
import json
from skimage import io

from maskToPolygons import MaskToPolygons
import utils

categories = utils.get_categories()

def createObjects(category_mask):
    converter = MaskToPolygons()
    categoryToPolygons, debug = converter.process(category_mask)

    objects = []
    counter = 0
    for cat in categoryToPolygons:
        for polygon in categoryToPolygons[cat]:
            obj = {}
            obj["name"] = categories[cat]
            obj["polygon"] = polygon
            obj["id"] = counter
            counter += 1
            objects.append(obj)
    return objects

def condense(string):
    new_string = ""
    for line in string.split("\n"):
        if any(c in line for c in ['{', '}', ':']):
            new_string += "\n" + line
        else:
            new_string += line.strip()
    return new_string

parser = argparse.ArgumentParser()
parser.add_argument("-p", help="Project name")
args = parser.parse_args()

config = None
with open("../LabelMe/data_config.json", 'r') as f:
    data_config = json.load(f)
    config = data_config[args.p]

root_category_mask = os.path.join(config["pspnet_prediction"],"category_mask")
# root_category_mask = config["ground_truth"]
root_polygons = config["polygons"]
im_list_path = config["im_list"]

im_list = []
if im_list_path:
    im_list = [line.rstrip() for line in open(im_list_path, 'r')]
else:
    im_list = [f.replace(".png", ".jpg") for f in os.listdir(root_category_mask) if ".png" in f]

for im in im_list:
    category_mask_name = im.replace(".jpg", ".png")
    polygon_file = im.replace(".jpg", "-polygons.json")
    category_mask_path = os.path.join(root_category_mask, category_mask_name)

    if os.path.exists(category_mask_path):
        print im
        category_mask = io.imread(category_mask_path, as_grey=True)
        objects = createObjects(category_mask)

        data = {}
        data["filename"] = im
        data["folder"] = args.p
        data["objects"] = objects

        polygon_file_path = os.path.join(root_polygons, polygon_file)
        if not os.path.exists(os.path.dirname(polygon_file_path)):
            os.makedirs(os.path.dirname(polygon_file_path))
        with open(polygon_file_path, 'w') as f:
            string = json.dumps(data,indent=2, sort_keys=True)
            string = condense(string)
            f.write(string)

