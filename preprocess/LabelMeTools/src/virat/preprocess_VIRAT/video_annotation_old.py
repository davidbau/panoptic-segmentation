import os
import numpy as np
from collections import defaultdict

class VideoAnnotation:

    def __init__(self, vid_name, ann_dir):
        events_fn = os.path.join(ann_dir, vid_name + ".viratdata.events.txt")
        objects_fn = os.path.join(ann_dir, vid_name + ".viratdata.objects.txt")
        mapping_fn = os.path.join(ann_dir, vid_name + ".viratdata.mapping.txt")
        if not os.path.exists(events_fn) or not os.path.exists(objects_fn) or not os.path.exists(mapping_fn):
            self.loaded = False
            return

        self.vid_name = vid_name
        self.events_list = read_file(events_fn)
        self.objects_list = read_file(objects_fn)
        self.mapping_list = read_file(mapping_fn)

        self.objCats = {}
        self.eventCats = {}
        self.writeCats()

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
            copy["category_id"] = obj["category_id"]
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
            copy["category_id"] = evt["category_id"]
            copy["name"] = evt["name"]
            copy["bbox"] = evt["bbox"][frame]
            copy["object_ids"] = evt["object_ids"]
            copies.append(copy)
        return copies

    def getObjectName(self, category_id):
        if category_id in self.objCats:
            return self.objCats[category_id]
        else:
            return "null"

    def getEventName(self, category_id):
        if category_id in self.eventCats:
            return self.eventCats[category_id]
        else:
            return "null"

    def writeCats(self):
        self.objCats[1] = "person"
        self.objCats[2] = "car"
        self.objCats[3] = "vehicle"
        self.objCats[4] = "object"
        self.objCats[5] = "bike"

        self.eventCats[1] = "loading_object_to_vehicle"
        self.eventCats[2] = "unloading_object_from_vehicle"
        self.eventCats[3] = "open_car_trunk"
        self.eventCats[4] = "close_car_trunk"
        self.eventCats[5] = "get_into_car"
        self.eventCats[6] = "get_out_of_car"
        self.eventCats[7] = "gesturing"
        self.eventCats[8] = "digging"
        self.eventCats[9] = "carrying"
        self.eventCats[10] = "running"
        self.eventCats[11] = "entering_facility"
        self.eventCats[12] = "exiting_facility"

    def createIndex(self):
        # Create objects index
        for line in self.objects_list:
            split = [int(x) for x in line.split()]
            id = split[0]
            duration = split[1]
            frame = split[2]
            bbox = split[3:7]
            category_id = split[7]

            if id in self.objects:
                obj = self.objects[id]
                obj["bbox"][frame] = bbox
            else:
                obj = {}
                obj["id"] = id
                obj["category_id"] = category_id
                obj["name"] = self.getObjectName(category_id)
                obj["duration"] = duration
                obj["bbox"] = {}
                obj["bbox"][frame] = bbox
                self.objects[id] = obj

                if type(obj["bbox"]) == int:
                    print "WHAT", obj["bbox"]

            self.frameToObjects[frame].append(id)

        # Create events index
        for line in self.events_list:
            split = [int(x) for x in line.split()]
            id = split[0]
            category_id = split[1]
            duration = split[2]
            start = split[3]
            end = split[4]
            frame = split[5]
            bbox = split[6:10]

            if id in self.events:
                event = self.events[id]
                event["bbox"][frame] = bbox
            else:
                event = {}
                event["id"] = id
                event["category_id"] = category_id
                event["name"] = self.getEventName(category_id)
                event["duration"] = duration
                event["start_frame"] = start
                event["end_frame"] = end
                event["bbox"] = {}
                event["bbox"][frame] = bbox
                self.events[id] = event

            self.frameToEvents[frame].append(id)

        # Create mapping index
        for line in self.mapping_list:
            split = [int(x) for x in line.split()]
            eid = split[0]
            category_id = split[1]
            duration = split[2]
            start = split[3]
            end = split[4]
            num_objs = split[5]
            cols = split[6:]
            oids = np.nonzero(cols)[0]

            self.events[eid]["object_ids"] = list(oids)


def read_file(fn):
    with open(fn,'r') as f:
        vid_list = f.read().splitlines()
        return vid_list

if __name__ == "__main__":
    VID_LIST = "./data/VIRAT/docs/list_release2.0.txt"
    ANN_DIR = "./data/VIRAT/annotations"

    vid_list = read_file(VID_LIST)
    for vid_name in vid_list:
        print vid_name
        vid_ann = VideoAnnotation(vid_name, ANN_DIR)

