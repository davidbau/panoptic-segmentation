import argparse
import os
import json
import numpy as np
from collections import defaultdict

class VideoAnnotation:

    def __init__(self, ann_fn):
        print("Loading ", ann_fn)
        self.basename = os.path.basename(ann_fn)
        self.vid_name = self.basename.replace(".json", "")

        if not os.path.exists(ann_fn):
            self.loaded = False
            return
        self.annotations = read_file(ann_fn)

        self.objects = {}
        self.events = {}
        self.frameToObjects = defaultdict(list)
        self.frameToEvents = defaultdict(list)
        self.createIndex()

        self.loaded = True

    def get_objects_at_frame(self, frame):
        ids = self.frameToObjects[frame]
        objs = [self.objects[id] for id in ids]

        # Copy objs with only one frame
        copies = []
        for obj in objs:
            copy = {}
            copy["id"] = obj["id"]
            copy["name"] = obj["name"]
            copy["bbox"] = obj["bbox"][frame]
            copies.append(copy)
        return copies

    def get_events_at_frame(self, frame):
        ids = self.frameToEvents[frame]
        evts = [self.events[id] for id in ids]

        # Copy objs with only one frame
        copies = []
        for evt in evts:
            copy = {}
            copy["id"] = evt["id"]
            copy["name"] = evt["name"]
            copy["bbox"] = evt["bbox"][frame]
            copy["object_ids"] = evt["object_ids"]
            copies.append(copy)
        return copies

    def createIndex(self):
        activities = self.annotations["activities"]
        objId = 0
        for a in activities:
            evt = {}
            evt["id"] = a["activityID"]
            evt["name"] = a["activity"]
            evt["bbox"] = {}
            evt["object_ids"] = []
            loc = a["localization"][self.vid_name + ".mp4"]
            for frame_num in loc:
                if loc[frame_num] == 1:
                    evt["start"] = int(frame_num)
                if loc[frame_num] == 0:
                    evt["end"] = int(frame_num)

            objects = a["objects"]
            for o in objects:
                obj = {}
                obj["id"] = objId
                objId += 1
                obj["name"] = o["objectType"]
                obj["bbox"] = {}
                loc = o["localization"][self.vid_name + ".mp4"]

                frames = [int(x) for x in loc.keys()]
                frames.sort()
                frame = frames[0]
                bbox = []
                while frame < evt["end"]:
                    if str(frame) in loc:
                        if "boundingBox" in loc[str(frame)]:
                            bbox = loc[str(frame)]["boundingBox"]
                            bbox = [bbox['x'], bbox['y'], bbox['w'], bbox['h']]
                        else:
                            break
                    obj["bbox"][frame] = bbox

                    # Create event bbox
                    if frame in evt["bbox"]:
                        evt["bbox"][frame] = bbox_union(evt["bbox"][frame], bbox)
                    else:
                        evt["bbox"][frame] = bbox

                    self.frameToObjects[frame].append(obj["id"])
                    self.frameToEvents[frame].append(evt["id"])
                    frame += 1

                evt["object_ids"].append(obj["id"])
                self.objects[obj["id"]] = obj
            self.events[evt["id"]] = evt

    def expandEventBboxes(self):
        # Looks better when visualized
        for id in self.events:
            evt = self.events[id]
            for frame in evt["bbox"]:
                x,y,w,h = evt["bbox"][frame]
                evt["bbox"][frame] = [x-10, y-10, w+20, h+20]

def bbox_union(bbox0, bbox1):
    x0,y0,w,h = bbox0
    x1 = x0 + w
    y1 = y0 + h
    a0,b0,w,h = bbox1
    a1 = a0 + w
    b1 = b0 + h

    x0 = min(x0, a0)
    y0 = min(y0, b0)
    x1 = max(x1, a1)
    y1 = max(y1, b1)
    return [x0, y0, x1-x0, y1-y0]

def read_file(file_name):
    with open(file_name, 'r') as f:
        dic = json.load(f)
        return dic

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-i', '--indir', type=str, default="../data/virat/raw_data/VIRAT-V1_JSON_validate-leaderboard_drop4_20180614/")
    parser.add_argument('-o', '--outdir', type=str, default="../data/virat/")
    args = parser.parse_args()

    file_index = read_file(os.path.join(args.indir, "file-index.json"))
    vid_list = [os.path.splitext(k)[0] for k in file_index]
    vid_list.sort()

    for vid_name in vid_list:
        ann_fn = os.path.join(args.indir, vid_name + ".json")
        vid_ann = VideoAnnotation(ann_fn)
        print("{}: {} objects".format(vid_ann.vid_name, len(vid_ann.objects)))
        print("{}: {} events".format(vid_ann.vid_name, len(vid_ann.events)))


