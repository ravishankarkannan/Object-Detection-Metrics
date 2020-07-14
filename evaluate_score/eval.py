import _init_paths
from BoundingBox import BoundingBox
from BoundingBoxes import BoundingBoxes
from Evaluator import *
from utils import *
import yaml
import glob
import os
import numpy as np
from argparse import ArgumentParser

class_map = {"0": "car", "1": "bicycle", "2": "person", "3": "sign"}


def getBoundingBoxes(parameter_config, args):
    """Read txt files containing bounding boxes (ground truth and detections)."""
    allBoundingBoxes = BoundingBoxes()
    # Read ground truths
    # currentPath = os.path.dirname(os.path.abspath(__file__))
    folderGT = os.path.join(args.data_file_path, 'groundtruths')
    os.chdir(folderGT)
    files = glob.glob("*.txt")
    files.sort()
    # Class representing bounding boxes (ground truths and detections)
    allBoundingBoxes = BoundingBoxes()
    # Read GT detections from txt file
    # Each line of the files in the groundtruths folder represents a ground truth bounding box
    # (bounding boxes that a detector should detect)
    # Each value of each line is  "class_id, x, y, width, height" respectively
    # Class_id represents the class of the bounding box
    # x, y represents the most top-left coordinates of the bounding box
    # x2, y2 represents the most bottom-right coordinates of the bounding box
    print(len(files))
    car_image_size_threshold = parameter_config["car_image_size_threshold"]
    person_image_size_threshold = parameter_config["person_image_size_threshold"]
    box_count = 0
    area = []
    for f in files:
        nameOfImage = f.replace(".txt", "")
        fh1 = open(f, "r")
        for line in fh1:
            line = line.replace("\n", "")
            if line.replace(' ', '') == '':
                continue
            splitLine = line.split(" ")
            idClass = splitLine[0]  # class
            x = float(splitLine[1])  # confidence
            y = float(splitLine[2])
            xmax = float(splitLine[3])
            ymax = float(splitLine[4])
            w = xmax - x
            h = ymax - y
            try:
                area.append(int(w) * int(h))
            except:
                print(nameOfImage, w, h)
            if class_map[idClass] == "car" and w * h < car_image_size_threshold:
                continue
            elif class_map[idClass] == "person" and w * h < person_image_size_threshold:
                continue

            bb = BoundingBox(
                nameOfImage,
                idClass,
                x,
                y,
                w,
                h,
                CoordinatesType.Absolute, (parameter_config["width"], parameter_config["height"]),
                BBType.GroundTruth,
                format=BBFormat.XYWH)
            box_count += 1
            allBoundingBoxes.addBoundingBox(bb)
        fh1.close()
    print("GroundTruth boxes count %s", box_count)
    # Read detections
    folderDet = os.path.join(args.data_file_path, 'detections')
    os.chdir(folderDet)
    files = glob.glob("*.txt")
    files.sort()
    # Read detections from txt file
    # Each line of the files in the detections folder represents a detected bounding box.
    # Each value of each line is  "class_id, confidence, x, y, width, height" respectively
    # Class_id represents the class of the detected bounding box
    # Confidence represents confidence (from 0 to 1) that this detection belongs to the class_id.
    # x, y represents the most top-left coordinates of the bounding box
    # x2, y2 represents the most bottom-right coordinates of the bounding box
    print(len(files))
    box_count = 0
    for f in files:
        # nameOfImage = f.replace("_det.txt","")
        nameOfImage = f.replace(".txt", "")
        # Read detections from txt file
        fh1 = open(f, "r")
        for line in fh1:
            line = line.replace("\n", "")
            if line.replace(' ', '') == '':
                continue
            splitLine = line.split(" ")
            idClass = splitLine[0]  # class
            confidence = float(splitLine[1])  # confidence
            x = float(splitLine[2])
            y = float(splitLine[3])
            xmax = float(splitLine[4])
            ymax = float(splitLine[5])
            w = xmax - x
            h = ymax - y
            if class_map[idClass] == "car" and w * h < car_image_size_threshold:
                continue
            elif class_map[idClass] == "person" and w * h < person_image_size_threshold:
                continue
            bb = BoundingBox(
                nameOfImage,
                idClass,
                x,
                y,
                w,
                h,
                CoordinatesType.Absolute, (parameter_config["width"], parameter_config["height"]),
                BBType.Detected,
                confidence,
                format=BBFormat.XYWH)
            box_count += 1
            allBoundingBoxes.addBoundingBox(bb)
        fh1.close()
    print("Detection boxes count %s", box_count)
    return allBoundingBoxes


def createImages(dictGroundTruth, dictDetected):
    """Create representative images with bounding boxes."""
    import numpy as np
    import cv2
    # Define image size
    width = 200
    height = 200
    # Loop through the dictionary with ground truth detections
    for key in dictGroundTruth:
        image = np.zeros((height, width, 3), np.uint8)
        gt_boundingboxes = dictGroundTruth[key]
        image = gt_boundingboxes.drawAllBoundingBoxes(image)
        detection_boundingboxes = dictDetected[key]
        image = detection_boundingboxes.drawAllBoundingBoxes(image)
        # Show detection and its GT
        cv2.imshow(key, image)
        cv2.waitKey()


def get_parameters(config_file):
    with open(config_file, 'rb') as file:
        parameter_config = yaml.load(file.read(), Loader=yaml.FullLoader)
        return parameter_config


def main(args):
    config_file_path = args.config
    data_file_path = args.data_file_path
    parameter_config = get_parameters(config_file_path)
    iouThreshold = parameter_config["iouThreshold"]
    car_image_size_threshold = parameter_config["car_image_size_threshold"]
    person_image_size_threshold = parameter_config["person_image_size_threshold"]
    # Read txt files containing bounding boxes (ground truth and detections)
    boundingboxes = getBoundingBoxes(parameter_config, args)
    # Uncomment the line below to generate images based on the bounding boxes
    # createImages(dictGroundTruth, dictDetected)
    # Create an evaluator object in order to obtain the metrics
    evaluator = Evaluator()
    ##############################################################
    # VOC PASCAL Metrics
    ##############################################################
    # Plot Precision x Recall curve
    evaluator.PlotPrecisionRecallCurve(
        boundingboxes,  # Object containing all bounding boxes (ground truths and detections)
        IOUThreshold=iouThreshold,  # IOU threshold
        method=MethodAveragePrecision.EveryPointInterpolation,  # As the official matlab code
        showAP=True,  # Show Average Precision in the title of the plot
        showInterpolatedPrecision=True,
        car_image_size_threshold=car_image_size_threshold,
        person_image_size_threshold=person_image_size_threshold)  # Plot the interpolated precision curve
    # Get metrics with PASCAL VOC metrics
    metricsPerClass = evaluator.GetPascalVOCMetrics(
        boundingboxes,  # Object containing all bounding boxes (ground truths and detections)
        IOUThreshold=iouThreshold,  # IOU threshold
        method=MethodAveragePrecision.EveryPointInterpolation)  # As the official matlab code
    print("Average precision values per class:\n")
    # Loop through classes to obtain their metrics
    for mc in metricsPerClass:
        # Get metric values per each class
        c = mc['class']
        precision = mc['precision']
        recall = mc['recall']
        average_precision = mc['AP']
        ipre = mc['interpolated precision']
        irec = mc['interpolated recall']
        # Print AP per class
        print('%s: %f' % (c, average_precision))


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument('-c', '--config',
                        default="/Users/ravikannan/Desktop/workspace/supporting_files/Object-Detection-Metrics/evaluate_score/config.yaml")
    parser.add_argument('-fp', '--data_file_path',
                        default="/Users/ravikannan/Desktop/workspace/supporting_files/Object-Detection-Metrics/evaluate_score")
    args = parser.parse_args()
    main(args)

# Modify the configuration file path and data file path.
# Provide the directory path to the folder containing "detections" and "groundtruths" folders.
# Make sure the detections are in the following format: <class> <class_confidence> <left> <top> <right> <bottom>
# Make sure the groundtruths are in the following format: <class> <left> <top> <right> <bottom>
