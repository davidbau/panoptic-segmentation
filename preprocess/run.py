# Creates first iteration of masks for ADE
import sys
import pdb
sys.path.append('./LabelMeTools/src/coco_utils')
sys.path.append('./LabelMeTools/src/ade20k')
import os
from glob import glob
import convert_ade20k_full as utils_ade
from coco_format import *
import json
from tqdm import tqdm
from multiprocessing import Pool

import PIL.Image as Image

from pycocotools import mask as COCOmask

map_to_ade = [
    0,
    2978,
    312,
    2420,
    976,
    2855,
    447,
    2131,
    165,
    3055,
    1125,
    350,
    2377,
    1831,
    838,
    [774, 783],
    2684,
    1610,
    1910,
    687,
    471,
    401,
    2994,
    1735,
    2473,
    2329,
    1276,
    2264,
    1564,
    2178,
    913,
    57,
    2272,
    907,
    724,
    2138,
    [2985, 533],
    1395,
    155,
    2053,
    689,
    137,
    266,
    581,
    2380,
    491,
    627,
    2212,
    2388,
    2423,
    943,
    2096,
    1121,
    1788,
    2530,
    2185,
    420,
    1948,
    1869,
    2251,
    2531,
    2128,
    294,
    239,
    212,
    571,
    2793,
    978,
    236,
    1240,
    181,
    629,
    2598,
    1744,
    1374,
    591,
    2679,
    223,
    123,
    47,
    1282,
    327,
    2821,
    1451,
    2880,
    2828,
    480,
    77,
    2616,
    [246, 247],
    2733,
    14,
    738,
    38,
    1936,
    1401,
    120,
    868,
    1702,
    249,
    308,
    1969,
    2526,
    2928,
    2337,
    1023,
    609,
    389,
    2989,
    1930,
    2668,
    2586,
    131,
    146,
    3016,
    2739,
    95,
    1563,
    642,
    1708,
    103,
    1002,
    2569,
    2704,
    2833,
    1551,
    1981,
    29,
    187,
    1393,
    747,
    2254,
    206,
    2262,
    1260,
    2243,
    2932,
    2836,
    2850,
    64,
    894,
    1858,
    3117,
    1919,
    1583,
    318,
    2356,
    2046,
    1098,
    530,
    954,

    451,
    3105,
    3101,
    1820,
    3102,
    982,
    3109,
    3108,
    3104,
    3110,
    [3103, 2293], # semi-flush mount light(s)
    1445,
    [2509, 2510, 3121], # spotlight, spot / spotlights / spotlight
    3118,
    2981,
    [970, 971], # floodlight(s)
]

ade_to_mapped = { }

is_thing = [
    False,
    False,
    False,
    False,
    False,
    True,
    False,
    False,
    True,
    True,
    False,
    True,
    False,
    True,
    False,
    True,
    True,
    False,
    True,
    True,
    True,
    True,
    False,
    True,
    True,
    True,
    False,
    False,
    True,
    False,
    False,
    True,
    True,
    True,
    True,
    True,
    True,
    True,
    True,
    True,
    True,
    True,
    True,
    True,
    True,
    True,
    True,
    False,
    True,
    False,
    True,
    True,
    False,
    False,
    True,
    False,
    True,
    True,
    True,
    True,
    False,
    False,
    False,
    True,
    True,
    True,
    True,
    True,
    True,
    False,
    True,
    True,
    True,
    True,
    True,
    True,
    True,
    True,
    True,
    True,
    False,
    True,
    True,
    True,
    True,
    False,
    True,
    True,
    True,
    True,
    True,
    True,
    False,
    True,
    True,
    False,
    True,
    False,
    True,
    True,
    True,
    True,
    False,
    True,
    True,
    True,
    False,
    True,
    True,
    True,
    False,
    True,
    True,
    True,
    False,
    True,
    True,
    True,
    True,
    True,
    True,
    True,
    True,
    True,
    True,
    True,
    True,
    True,
    True,
    False,
    True,
    True,
    True,
    True,
    True,
    True,
    True,
    True,
    True,
    True,
    True,
    False,
    True,
    True,
    True,
    True,
    True,
    True,
    True,
    True,
    True,

    True,
    True,
    True,
    True,
    True,
    True,
    True,
    True,
    True,
    True,
    True, # semi-flush mount light(s)
    True,
    True, # spotlight, spot / spotlights / spotlight
    True,
    True,
    True, # floodlight(s)
]

categories = zip(map_to_ade, is_thing)
categories = sorted(categories, key=lambda x: x[1])

map_to_ade, is_thing = zip(*categories)

for i, category in enumerate(map_to_ade):
    if type(category) is list:
        for id in category:
            ade_to_mapped[id] = i
    else:
        ade_to_mapped[category] = i
        # print(str(category) + ' -> ' + str(i))

# print(ade_to_mapped)

def get_simplified_category(ade_cat):
    return ade_to_mapped[ade_cat] if ade_cat in ade_to_mapped else 0

def get_simplified_category_name(sim_cat):
    ade_categories = map_to_ade[sim_cat]
    if not type(ade_categories) is list:
        ade_categories = [ade_categories]
    return ' - '.join([cat_list[c] for c in ade_categories])

def make_annotation(input):
    i, ann_dir, im_name, as_gt = input
    ann_path = os.path.join(ann_dir, im_name).replace('.jpg', '_seg.png')

    # Parts not handled
    # parts_path = os.path.join(ann_dir, im_name).replace('.jpg', '_seg_parts_N.png')

    ann_image = cv2.imread(ann_path)
    if ann_image is None:
        print("Skipping", ann_path)
        return None

    ins_mask = ann_image[:, :, 0]  # B
    g_mask = ann_image[:, :, 1]  # G
    r_mask = ann_image[:, :, 2]  # R
    cat_mask = (r_mask.astype(int) / 10) * 256 + g_mask

    mapped_sem_image = np.zeros_like(cat_mask)
    for cat in np.unique(cat_mask.astype(int)):
        mask = (cat_mask == cat)
        mapped_sem_image[mask] = get_simplified_category(cat)
    ann_image_path = os.path.join(ann_dir, im_name).replace('images', 'annotations').replace('.jpg', '.png')
    if not os.path.exists(os.path.dirname(ann_image_path)):
        try:
            os.makedirs(os.path.dirname(ann_image_path))
        except FileExistsError:
            pass
    Image.fromarray(mapped_sem_image).convert('L').save(ann_image_path)

    # annotation_json_path = os.path.join(ann_dir, im_name).replace('.jpg', '.json')
    # with open(annotation_json_path) as annotation_file:
    #     annotation_data = json.load(annotation_file)['annotation']
    #
    # assert annotation_data['imsize'][0] == ann_image.shape[0]
    # assert annotation_data['imsize'][1] == ann_image.shape[1]
    #
    # obj_index = {}
    # for obj in annotation_data['object']:
    #     obj_index[obj['id']] = obj

    annotations = []
    for ins in np.unique(ins_mask):
        if ins == 0:
            continue
        mask = (ins_mask == ins)
        cat = np.sum(cat_mask[mask]) / np.sum(mask)
        crowd = 0

        simp_cat = get_simplified_category(int(cat))
        if not is_thing[simp_cat]:
            continue # skip stuff categories

        contours_mask = np.zeros_like(ins_mask, dtype=np.uint8)
        contours_mask[mask] = 1
        contours, hierarchy = cv2.findContours(contours_mask, cv2.RETR_EXTERNAL,
                                                         cv2.CHAIN_APPROX_SIMPLE)
        polygons = []
        for contour in contours:
            contour_flattened = contour.flatten().tolist()
            if len(contour_flattened) > 4: # more than 2 vertices (=2*2 coordinates)
                polygons.append(contour_flattened)

        if len(polygons) == 0:
            continue # skip if we couldn't find any polygons

        ann = make_ann(mask, iscrowd=crowd)
        ann["segmentation"] = polygons
        ann["image_id"] = i + 1
        ann["category_id"] = simp_cat
        ann["id"] = len(annotations) + 1
        ann["method"] = {"name": "GT"}
        ann["iteration"] = 0
        annotations.append(ann)

    image_info = {}
    image_info["height"] = ann_image.shape[0]
    image_info["width"] = ann_image.shape[1]
    image_info["path"] = im_name
    if as_gt:
        image_info["gt_annotations"] = annotations
    else:
        image_info["annotations"] = annotations
    return image_info


# def make_annotation(input):
#     i, ann_dir, im_name, as_gt = input
#     ann_path = os.path.join(ann_dir, im_name).replace('.jpg', '_seg.png')
#
#     # Parts not handled
#     # parts_path = os.path.join(ann_dir, im_name).replace('.jpg', '_seg_parts_N.png')
#
#     ann_image = cv2.imread(ann_path)
#     if ann_image is None:
#         print("Skipping", ann_path)
#         return None
#
#     ins_mask = ann_image[:, :, 0]  # B
#     g_mask = ann_image[:, :, 1]  # G
#     r_mask = ann_image[:, :, 2]  # R
#     cat_mask = (r_mask.astype(int) / 10) * 256 + g_mask
#
#     mapped_sem_image = np.zeros_like(cat_mask)
#     for cat in np.unique(cat_mask.astype(int)):
#         mask = (cat_mask == cat)
#         mapped_sem_image[mask] = get_simplified_category(cat)
#     ann_image_path = os.path.join(ann_dir, im_name).replace('images', 'annotations').replace('.jpg', '.png')
#     if not os.path.exists(os.path.dirname(ann_image_path)):
#         try:
#             os.makedirs(os.path.dirname(ann_image_path))
#         except FileExistsError:
#             pass
#     Image.fromarray(mapped_sem_image).convert('L').save(ann_image_path)
#
#     annotation_json_path = os.path.join(ann_dir, im_name).replace('.jpg', '.json')
#     with open(annotation_json_path) as annotation_file:
#         annotation_data = json.load(annotation_file)['annotation']
#
#     assert annotation_data['imsize'][0] == ann_image.shape[0]
#     assert annotation_data['imsize'][1] == ann_image.shape[1]
#
#     annotations = []
#     for object in annotation_data['object']:
#         simp_cat = get_simplified_category(object['namendx'])
#         if not is_thing[simp_cat]:
#             continue # skip stuff
#
#         ann_polygon = object['polygon']
#
#         polygon = []
#         for x, y in zip(ann_polygon['x'], ann_polygon['y']):
#             polygon.append(x)
#             polygon.append(y)
#         coco_polygon = [polygon]
#
#         ann = {}
#         ann["segmentation"] = coco_polygon
#         ann["bbox"] = list(COCOmask.toBbox(segm))
#         ann["area"] = int(COCOmask.area(segm))
#         ann["iscrowd"] = int(iscrowd)
#         ann["image_id"] = i + 1
#         ann["category_id"] = simp_cat
#         return ann
#
#     image_info = {}
#     image_info["height"] = ann_image.shape[0]
#     image_info["width"] = ann_image.shape[1]
#     image_info["path"] = im_name
#     if as_gt:
#         image_info["gt_annotations"] = annotations
#     else:
#         image_info["annotations"] = annotations
#     return image_info


def make_annotations(ann_dir, im_list, as_gt=False, multi_proc=True):
    images_annotated = []
    if multi_proc:
        data_pairs = [[i, ann_dir, im_name, as_gt] for i, im_name in enumerate(im_list)]
        pool = Pool(processes=os.cpu_count())
        pool = Pool(processes=1)
        images_annotated = pool.map(make_annotation, data_pairs)
    else:
        for i, im_name in enumerate(tqdm(im_list)):
            image_info = make_annotation(ann_dir, im_name, as_gt)
            images_annotated.append(image_info)
    return images_annotated


if __name__ == "__main__":

    ann_dir = '/data/vision/torralba/datasets/ADE20K_2020_04_01'
    anno_path = '/data/vision/torralba/datasets/ADE20K_2020_04_01/annotations'
    # datasets = ['frames_training', 'places_training']
    datasets = ['places_validation', 'frames_training', 'places_training']
    # datasets = ['bframes_training']
    multi_proc = True

    contents = utils_ade.open_mat_file(ann_dir)
    cat_list = contents[2]
    cat_list.insert(0, '__background__')

    print('Using categories:')    
    num_is_thing = 0
    for i in range(len(map_to_ade)):
        print(str(i) + ': ' + get_simplified_category_name(i))            
        if is_thing[i]:
            num_is_thing += 1
    print('# thing classes: ' + str(num_is_thing))
    print('# stuff classes: ' + str(len(map_to_ade) - num_is_thing))

    for dataset in datasets:
        print('Processing dataset "{}"'.format(dataset))
        image_paths = [os.path.relpath(y, ann_dir) for x in os.walk(ann_dir + '/images/' + dataset) for y in glob(os.path.join(x[0], '*.jpg'))]
        print("# images: {}".format(len(image_paths)))

        with open('{}/output_images.json'.format(anno_path), 'w+') as f:
            json.dump(image_paths, f, indent=4)


        image_annotations = make_annotations(ann_dir, image_paths, as_gt=False, multi_proc=multi_proc)
        # with open('{}/output_annotations.json'.format(anno_path), 'w+') as f:
        #     json.dump(image_annotations, f, indent=4)

        images = []
        annotations = []
        for i, image_info in enumerate(image_annotations):
            image_name_full = image_paths[i]
            images.append({
                'file_name': os.path.relpath(os.path.join(ann_dir, image_name_full), ann_dir + '/images/' + dataset),
                'height': image_info['height'],
                'width': image_info['width'],
                'id': i + 1,
            })
            for annotation in image_info['annotations']:
                ann = annotation.copy()
                ann['id'] = len(annotations) + 1
                annotations.append(ann)

        categories = []
        for i in range(len(map_to_ade)):
            if not is_thing[i]:
                continue
            categories.append({
                'id': i,
                'name': get_simplified_category_name(i),
                'isthing': 1 if is_thing[i] else 0,
            })

        # for i in range(len(map_to_ade)):
        #     ade_categories = map_to_ade[i]
        #     if not type(ade_categories) is list:
        #         ade_categories = [ade_categories]
        #     print(str(i) + ' -> ' + ' - '.join([cat_list[c] for c in ade_categories]) + ' ; isthing=' + str(is_thing[i]))

        with open('{}/output_images.json'.format(anno_path), 'w+') as f:
            json.dump(images, f, indent=4)

        with open('{}/output_annotations.json'.format(anno_path), 'w+') as f:
            json.dump(annotations, f, indent=4)

        with open('./categories.json', 'w+') as f:
            json.dump(categories, f, indent=4)

        with open('{}/instances_{}_gts.json'.format(anno_path, dataset), 'w+') as f:
            json.dump({
                'images': images,
                'annotations': annotations,
                'categories': categories,
            }, f, indent=4)

        num_things = 0
        for i in range(len(is_thing)):
            if is_thing[i]:
                num_things += 1

    print('# Thing Classes: {}'.format(num_things))
    print('# Stuff Classes: {}'.format(len(is_thing) - num_things))

    # for i, annotation in enumerate(annotations):
    #     image_name_full = images[i].replace('.jpg', '.json')
    #     file_name = os.path.join(anno_path, 'split_{}'.format(split), image_name_full)
    #     folder_name = os.path.dirname(file_name)
    #
    #     if not os.path.isdir(folder_name):
    #         os.makedirs(folder_name)
    #     with open(file_name, 'w+') as f:
    #         json.dump(annotation, f, indent=4)
