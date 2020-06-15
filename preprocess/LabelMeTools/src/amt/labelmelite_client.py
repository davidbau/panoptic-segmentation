import os
import argparse
import requests
import json
from tqdm import tqdm

from bundle_utils import *

def get_request(endpoint):
    r = requests.get(url=endpoint)
    if r.status_code != 200:
        print(r, endpoint)
        return None
    return json.loads(r.text)

class LabelMeLiteClient:

    def __init__(self, local=False):
        self.base_url = "https://labelmelite.csail.mit.edu"
        # Edit https config to add results endpoint
        self.results_url = "http://vision01.csail.mit.edu:3000" 
        if local:
            self.base_url = "http://localhost:3000"

        self.bundle_size = 30

    # 
    # Get Bundles
    # 
    def get_bundle(self, job_id, bundle_id):
        endpoint = self.base_url + "/bundles/" + job_id + "/" + bundle_id + ".json"
        bundle = get_request(endpoint)
        return bundle
    def get_bundle_ids(self, job_id):
        endpoint = self.base_url + "/api/bundles?job_id=" + job_id;
        bundle_ids = get_request(endpoint)
        if bundle_ids is None:
            bundle_ids = []
        print("Retrieved {} bundle_ids for {}".format(len(bundle_ids), job_id))
        return bundle_ids
    def get_bundles(self, job_id):
        bundles = []
        bundle_ids = self.get_bundle_ids(job_id)
        for bundle_id in bundle_ids:
            bundle = self.get_bundle(job_id, bundle_id)
            if bundle:
                bundles.append(bundles)
        print("Retrieved {} bundles for {}".format(len(bundle_ids), job_id))
        return bundles

    # 
    # Post Bundles
    # 
    def post_bundles(self, job_id, coco):
        endpoint = self.base_url + "/api/bundles?job_id=" + job_id;
        bundles = split_bundles(coco, bundle_size=self.bundle_size)

        print("Posting {} bundles to {}".format(len(bundles), endpoint))
        bundle_ids = []
        for bundle in tqdm(bundles):
            r = requests.post(url=endpoint, json=bundle)
            if r.status_code != 200:
                print("Response", r)
                continue
            res = json.loads(r.text)
            bundle_ids.append(res["bundle_id"])
        print("Successfully posted {} bundles".format(len(bundle_ids)))

    # 
    # Get Results
    # 
    def get_result(self, job_id, result_id):
        endpoint = self.results_url + "/results/" + job_id + "/" + result_id + ".json"
        result = get_request(endpoint)
        return result
    def get_result_ids(self, job_id):
        endpoint = self.results_url + "/api/results?job_id=" + job_id;
        result_ids = get_request(endpoint)
        if result_ids is None:
            result_ids = []
        print("Retrieved {} result_ids for {}".format(len(result_ids), job_id))
        return result_ids
    def get_results(self, job_id):
        results = []
        result_ids = self.get_result_ids(job_id)
        for result_id in result_ids:
            result = self.get_result(job_id, result_id)
            if result:
                results.append(result)
        print("Retrieved {} results for {}".format(len(results), job_id))
        return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-j', '--job_id', type=str, required=True)
    parser.add_argument('-f', '--ann_fn', type=str)
    parser.add_argument('-l', '--local', action='store_true')
    args = parser.parse_args()
    print(args)

    lml_client = LabelMeLiteClient(args.local)

    if args.ann_fn:
        coco = COCO(args.ann_fn)
        lml_client.post_bundles(args.job_id, coco)
    else:
        bundle_ids = lml_client.get_bundles(args.job_id)



