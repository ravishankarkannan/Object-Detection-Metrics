import numpy as np
import glob
import os
from argparse import ArgumentParser
import json
import untangle
from collections import Counter

file_type = "txt"


def load_json(file_path):
    with open(file_path) as data_file:
        data = json.load(data_file)
    return data


def main(args):
    file_path = args.file_path
    vehicle_count = Counter()
    type_count = Counter()
    color_count = Counter()
    if file_type == "json":
        data = load_json(file_path)
    elif file_type == "xml":
        data = untangle.parse(file_path)
        for image in data.annotations.image:
            if len(image) > 0:
                for box in image.box:
                    vehicle_count[box['label']] += 1
                    if box['label'] == "four+_wheel":
                        for attribute in box.attribute:
                            if attribute['name'] == "type":
                                type_count[attribute.cdata] += 1
                            if attribute['name'] == "color":
                                color_count[attribute.cdata] += 1
    elif file_type == "txt":
        folderGT = os.path.join(file_path)
        os.chdir(folderGT)
        files = glob.glob(folderGT+"/*.txt")
        print(len(files))
        box_count = 0
    print("vehicle_count", vehicle_count)
    print("type_count", type_count)
    print("color_count", color_count)
    return 0


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument('-fp', '--file_path',
                        default='/Users/ravikannan/Desktop/workspace/supporting_files/Object-Detection-Metrics/script/annotations.xml')
    args = parser.parse_args()
    main(args)
