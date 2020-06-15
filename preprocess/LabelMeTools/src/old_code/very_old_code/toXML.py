import os
import json
import time

def makePolygonTag(obj):
    polygon = obj["polygon"]
    polygon_body = "<username/>\n"
    for p in polygon:
        x_tag = "<x>{}</x>".format(p[0])
        y_tag = "<y>{}</y>".format(p[1])
        p_tag = "<pt>\n{}\n{}\n</pt>\n".format(x_tag,y_tag)

        polygon_body += p_tag
    polygon_tag = "<polygon>\n{}</polygon>".format(polygon_body)
    return polygon_tag


def makePartsTag(obj):
    body = ""
    if "parts" in obj:
        parts = obj["parts"]
        if parts["hasparts"]:
            body += "<hasparts>{}</hasparts>".format(parts["hasparts"][0])
        else:
            body += "<hasparts/>"

        if parts["ispartof"]:
            body += "<ispartof>{}</ispartof>".format(parts["ispartof"])
        else:
            body += "<ispartof/>"
    else:
        body = "<hasparts/>\n<ispartof/>"
    return "<parts>\n{}\n</parts>".format(body)

def toXML(data):
    filename_tag = "<filename>{}</filename>".format(data["filename"])
    folder_tag = "<folder>{}</folder>".format(data["folder"])
    source_tag = "<source>\n<sourceImage>The Help Movie</sourceImage>\n<sourceAnnotation>LabelMe Webtool</sourceAnnotation>\n</source>"
    body = "{}\n{}\n".format(filename_tag, folder_tag)

    for obj in data["objects"]:
        name_tag = "<name>{}</name>".format(obj["name"])
        deleted_tag = "<deleted>0</deleted>"
        verified_tag = "<verified>0</verified>"
        date_tag = "<date>{}</date>".format(time.asctime( time.localtime(time.time()) ))

        polygon_tag = makePolygonTag(obj)
        viewpoint_tag = "<viewpoint/>"
        id_tag = "<id>{}</id>".format(obj["id"])
        parts_tag = makePartsTag(obj)

        obj_tag = "{}\n{}\n{}\n{}\n{}\n{}\n{}\n{}\n".format(name_tag,deleted_tag,verified_tag,date_tag, polygon_tag,viewpoint_tag, id_tag, parts_tag)
        body += "<object>\n{}\n</object>".format(obj_tag)

    xml = "<annotation>\n{}</annotation>".format(body)
    return xml

polygons_root = "/data/vision/oliva/scenedataset/scaleplaces/movie/polygons/the_help_2011/"
xml_root = "/data/vision/oliva/scenedataset/scaleplaces/movie/xml_approx/the_help_2011/"
if not os.path.exists(xml_root):
    os.makedirs(xml_root)

for filename in os.listdir(polygons_root):
    if "-polygons.json" in filename:

        xml = ""
        with open(os.path.join(polygons_root,filename), 'r') as f1:
            data = json.load(f1)
            xml = toXML(data)

        xml_filename = filename.replace("-polygons.json", ".xml")
        xml_file = os.path.join(xml_root, xml_filename)
        
        with open(xml_file, 'w') as f2:
            f2.write(xml)
