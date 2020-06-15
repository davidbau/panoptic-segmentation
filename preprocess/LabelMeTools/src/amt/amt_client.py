import os
import argparse
from tqdm import tqdm

from boto.mturk.connection import MTurkConnection
from boto.mturk.question import ExternalQuestion
from boto.mturk.qualification import (Qualifications, 
    PercentAssignmentsApprovedRequirement, 
    NumberHitsApprovedRequirement)
from boto.mturk.price import Price

import amt_config
import hit_properties
from bundle_utils import *
from labelmelite_client import LabelMeLiteClient

class AMTClient:

    def __init__(self, sandbox=True):
        if sandbox:
            self.host = "mechanicalturk.sandbox.amazonaws.com"
            self.external_submit_endpoint = "https://workersandbox.mturk.com/mturk/externalSubmit"
        else:
            self.host = "mechanicalturk.amazonaws.com"
            self.external_submit_endpoint = "https://www.mturk.com/mturk/externalSubmit"

        self.base_url = "https://labelmelite.csail.mit.edu"
        self.connection = MTurkConnection(
            aws_access_key_id=amt_config.AWS_ACCESS_KEY_ID,
            aws_secret_access_key=amt_config.AWS_SECRET_ACCESS_KEY,
            host=self.host)

    def create_hit(self, job_id, bundle_id, hitType="yesno"):
        params_to_encode = {"job_id": job_id,
                            "bundle_id": bundle_id,
                            "host": self.external_submit_endpoint}
        encoded_url = encode_get_parameters(self.base_url + "/amt_{}".format(hitType), params_to_encode)
        # print(encoded_url)

        if hitType == "yesno":
            props = hit_properties.YesNoHitProperties
        elif hitType == "edit":
            props = hit_properties.EditHitProperties
        else:
            raise Exception("Hit type not implemented")

        create_hit_result = self.connection.create_hit(
            title=props["title"],
            description=props["description"],
            keywords=props["keywords"],
            duration=props["duration"],
            max_assignments=props["max_assignments"],
            question=ExternalQuestion(encoded_url, props["frame_height"]),
            reward=Price(amount=props["reward"]),
            # Determines information returned by certain API methods.
            response_groups=('Minimal', 'HITDetail'),
            qualifications=props["qualifications"])


def encode_get_parameters(baseurl, arg_dict):
    queryString = baseurl + "?"
    for indx, key in enumerate(arg_dict):
        queryString += str(key) + "=" + str(arg_dict[key])
        if indx < len(arg_dict)-1:
            queryString += "&"
    return queryString


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-j', '--job_id', type=str, required=True)
    parser.add_argument('--hit_type', type=str, default="yesno")
    # parser.add_argument('-p', '--prod', action='store_true')
    args = parser.parse_args()
    print(args)

    amt_client = AMTClient(sandbox=True)
    lml_client = LabelMeLiteClient()
    
    bundle_ids = lml_client.get_bundles(args.job_id)
    for bundle_id in tqdm(bundle_ids):
        amt_client.create_hit(args.job_id, bundle_id, hitType=args.hit_type)

