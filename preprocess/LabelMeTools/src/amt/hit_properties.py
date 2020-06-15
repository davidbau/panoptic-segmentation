from boto.mturk.qualification import (Qualifications, 
    PercentAssignmentsApprovedRequirement, 
    NumberHitsApprovedRequirement)

qualifications = Qualifications()
qual_1 = PercentAssignmentsApprovedRequirement(
    comparator="GreaterThan",
    integer_value="0")
# qual_2 = NumberHitsApprovedRequirement(
#     comparator="GreaterThan",
#     integer_value="0")
qualifications.add(qual_1)
# qualifications.add(qual_2)

YesNoHitProperties = {
  "title": "Choose good annotations of the following category. 5 categories",
  "description": "LabelMeLite Yes/No Tool",
  "keywords": "image,annotation",
  "reward": 0.05,
  "duration": 60*10,
  "frame_height": 800,
  "max_assignments": 5,
  "country": ["US", "DE"],
  "qualifications": qualifications
}


EditHitProperties = {
  "title": "Refine annotations of the following category",
  "description": "LabelMeLite Edit Tool",
  "keywords": "image,annotation",
  "reward": 0.05,
  "duration": 60*10,
  "frame_height": 800,
  "max_assignments": 5,
  "country": ["US", "DE"],
  "qualifications": qualifications
}