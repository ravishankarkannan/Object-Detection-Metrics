import os
import glob
import numpy as np
import sys
import argparse
from pathlib import Path
import cv2


class LabelConverter:
    def __init__(self, args):
        self.width = args.width
        self.height = args.height
        self.from_path = args.from_path
        self.to_path = args.to_path
        self.min_area = args.min_area
        self.video_writer = cv2.VideoWriter("maryland.avi", cv2.VideoWriter_fourcc(*'MJPG'), 30.0,
                                            (self.width, self.height))
        if self.video_writer.isOpened():
            print("Video Writer created")
        else:
            print("Video Writer cannot be created. Exiting...!")
            exit()
        # Create the output folder if not already
        print("image width :", self.width, " image height :", self.height)

    def remove_small_boxes(self, image, annotations):
        """Function to remove bboxes that are smaller than min area from annotations"""
        height, width, _ = image.shape
        image_area = height * width
        bbox_areas = annotations[:, 3] * annotations[:, 4] * image_area
        indices = list(np.where(bbox_areas > self.min_area)[0])
        return annotations[indices, :]

    def convert(self):
        """Function to convert all yolo format label files"""

        # Get all the CVAT YOLO Dataset folders exported from individual CVAT tasks
        folders = sorted(glob.glob(self.from_path + "/task*/"))
        # Create output folder to store converted label files and merged video
        Path(self.to_path).mkdir(exist_ok=True, parents=True)

        # Frame counter to rename label file to corresponding frame number
        frame_counter = 0
        for folder in folders:
            print(folder)
            # label files and images are stored in obj_train_data folder
            label_file_list = sorted(glob.glob(os.path.join(folder, 'obj_train_data', "*.txt")))

            if len(label_file_list) == 0:
                print("No label files were found!!")
                return 0

            for label_file in label_file_list:
                # Read the frame
                input_image_name = label_file.replace(".txt", ".jpg")
                image = cv2.imread(input_image_name)

                output_label_file_name = os.path.join(self.to_path, "frame_{:06d}.txt".format(frame_counter))
                frame_counter += 1

                # Load the annotation
                annotations = np.loadtxt(label_file)
                annotations = annotations.reshape((-1, 5))

                annotations = self.remove_small_boxes(image, annotations)
                # BBox coordinates are relative to image size. Scale them back to absolute numbers
                annotations[:, 1] *= self.width
                annotations[:, 3] *= self.width
                annotations[:, 2] *= self.height
                annotations[:, 4] *= self.height

                annotations[:, 1] -= annotations[:, 3] / 2
                annotations[:, 2] -= annotations[:, 4] / 2
                annotations[:, 3] += annotations[:, 1]
                annotations[:, 4] += annotations[:, 2]

                # annotations = annotations.astype(np.uint8)

                np.savetxt(output_label_file_name, annotations, fmt='%d')

                # resize images
                resized_image = cv2.resize(image, (self.width, self.height))
                # cv2.imwrite(output_image_name,resized_image)
                self.video_writer.write(resized_image)

        self.video_writer.release()
        # system_call = "ffmpeg -i " + self.video_file_name + " " + self.video_file_name.replace(".avi",".mp4")
        os.system("ffmpeg -i maryland.avi maryland.mp4")
        os.system("rm maryland.avi")


def get_args():
    """Function to set the arguments and parse them"""
    parser = argparse.ArgumentParser()
    parser.add_argument('--from_path', type=str, required=True, help=" Path to folder containing CVAT datasets")
    parser.add_argument('--to_path', type=str, required=True, help=" Path to write formated annotations")
    parser.add_argument('--width', type=int, required=True, help=" Image width to resize image ")
    parser.add_argument('--height', type=int, required=True, help=" Image height to resize image")
    parser.add_argument('--min_area', type=int, required=True, help="Minimum area for a bouding box")
    return parser.parse_args()


def main():
    args = get_args()
    # Get the list of YOLO Label formated annotation files
    label_converter = LabelConverter(args)
    label_converter.convert()


if __name__ == "__main__":
    main()

