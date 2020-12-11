# Script to convert yolo annotations to voc format
# Sample format
# <annotation>
#     <folder>_image_fashion</folder>
#     <filename>brooke-cagle-39574.jpg</filename>
#     <size>
#         <width>1200</width>
#         <height>800</height>
#         <depth>3</depth>
#     </size>
#     <segmented>0</segmented>
#     <object>
#         <name>head</name>
#         <pose>Unspecified</pose>
#         <truncated>0</truncated>
#         <difficult>0</difficult>
#         <bndbox>
#             <xmin>549</xmin>
#             <ymin>251</ymin>
#             <xmax>625</xmax>
#             <ymax>335</ymax>
#         </bndbox>
#     </object>
# <annotation>
import os
import xml.etree.cElementTree as ET
from PIL import Image
import argparse

def indent(elem, level=0):
     i = "\n" + level*"\t"
     if len(elem):
         if not elem.text or not elem.text.strip():
             elem.text = i + "\t"
         if not elem.tail or not elem.tail.strip():
             elem.tail = i
         for elem in elem:
             indent(elem, level+1)
         if not elem.tail or not elem.tail.strip():
             elem.tail = i
     else:
         if level and (not elem.tail or not elem.tail.strip()):
             elem.tail = i


def create_root(file_prefix, width, height):
    root = ET.Element("annotations")
    ET.SubElement(root, "folder").text = "images"
    ET.SubElement(root, "filename").text = "{}.jpg".format(file_prefix)
    ET.SubElement(root, "path").text = "{}.jpg".format(file_prefix)
    source = ET.SubElement(root, "source")
    ET.SubElement(source, "database").text = "{}.jpg".format('Unkonw')
    size = ET.SubElement(root, "size")
    ET.SubElement(size, "width").text = str(width)
    ET.SubElement(size, "height").text = str(height)
    ET.SubElement(size, "depth").text = "3"
    ET.SubElement(root, "segmented").text = "0"
    return root


def create_object_annotation(root, voc_labels):
    for voc_label in voc_labels:
        obj = ET.SubElement(root, "object")
        ET.SubElement(obj, "name").text = voc_label[0]
        ET.SubElement(obj, "pose").text = "Unspecified"
        ET.SubElement(obj, "truncated").text = str(0)
        ET.SubElement(obj, "difficult").text = str(0)
        bbox = ET.SubElement(obj, "bndbox")
        ET.SubElement(bbox, "xmin").text = str(int(voc_label[1]))
        ET.SubElement(bbox, "ymin").text = str(int(voc_label[2]))
        ET.SubElement(bbox, "xmax").text = str(int(voc_label[3]))
        ET.SubElement(bbox, "ymax").text = str(int(voc_label[4]))
        # obj = indent(obj,level=1)
    return root


def create_file(file_prefix, width, height, voc_labels):
    root = create_root(file_prefix, width, height)
    root = create_object_annotation(root, voc_labels)
    indent(root)
    tree = ET.ElementTree(root)
    tree.write("{}/{}.xml".format(DESTINATION_DIR, file_prefix))


def read_file(file_path):
    file_prefix = os.path.basename(file_path).split(".txt")[0]
    image_file_name = file_path.replace(".txt", ".jpg").replace("labels/", "")
    # image_file_name = "{}.jpg".format(file_prefix)
    # img = Image.open("{}/{}".format("images", image_file_name))
    img = Image.open(image_file_name)
    w, h = img.size
    with open(file_path, 'r') as file:
        lines = file.readlines()
        voc_labels = []
        for line in lines:
            voc = []
            line = line.strip()
            data = line.split()
            voc.append(CLASS_MAPPING.get(data[0]))
            bbox_width = float(data[3]) * w
            bbox_height = float(data[4]) * h
            center_x = float(data[1]) * w
            center_y = float(data[2]) * h
            voc.append(center_x - (bbox_width / 2))
            voc.append(center_y - (bbox_height / 2))
            voc.append(center_x + (bbox_width / 2))
            voc.append(center_y + (bbox_height / 2))
            voc_labels.append(voc)
        create_file(file_prefix, w, h, voc_labels)
    print("Processing complete for file: {}".format(file_path))


def start(dir_name):
    if not os.path.exists(DESTINATION_DIR):
        os.makedirs(DESTINATION_DIR)
    for filename in os.listdir(ANNOTATIONS_DIR_PREFIX):
        if filename.endswith('txt'):
            read_file(os.path.join(ANNOTATIONS_DIR_PREFIX, filename))
        else:
            print("Skipping file: {}".format(filename))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--image_folder", type=str, default="data/samples", help="path to dataset")
    ANNOTATIONS_DIR_PREFIX = "/mnt/disk84/yolov5/runs/detect/exp8/labels/"

    DESTINATION_DIR = "/mnt/disk84/yolov5/runs/detect/exp8/converted_labels"

    CLASS_MAPPING = {
        '0': 'wd1',
        '1': 'wd2',
        '2': 'bm',
        # Add your remaining classes here.
    }

    start('none')
