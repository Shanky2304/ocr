"""
Character Detection

The goal of this task is to implement an optical character recognition system consisting of Enrollment, Detection and Recognition sub tasks

Please complete all the functions that are labelled with '# TODO'. When implementing the functions,
comment the lines 'raise NotImplementedError' instead of deleting them.

Do NOT modify the code provided.
Please follow the guidelines mentioned in the project1.pdf
Do NOT import any library (function, module, etc.).
"""

import argparse
import json
import os
import glob
# opencv version - 4.5.4.60
import cv2
import csv
import numpy as np
from copy import deepcopy


def read_image(img_path, show=False):
    """Reads an image into memory as a grayscale array.
    """
    img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)

    if show:
        show_image(img)

    return img


def show_image(img, delay=1000):
    """Shows an image.
    """
    cv2.namedWindow('image', cv2.WINDOW_AUTOSIZE)
    cv2.imshow('image', img)
    cv2.waitKey(delay)
    cv2.destroyAllWindows()


def parse_args():
    parser = argparse.ArgumentParser(description="cse 473/573 project 1.")
    parser.add_argument(
        "--test_img", type=str, default="./data/test_img.jpg",
        help="path to the image used for character detection (do not change this arg)")
    parser.add_argument(
        "--character_folder_path", type=str, default="./data/characters",
        help="path to the characters folder")
    parser.add_argument(
        "--result_saving_directory", dest="rs_directory", type=str, default="./",
        help="directory to which results are saved (do not change this arg)")
    args = parser.parse_args()
    return args


def ocr(test_img, characters):
    """Step 1 : Enroll a set of characters. Also, you may store features in an intermediate file.
       Step 2 : Use connected component labeling to detect various characters in an test_img.
       Step 3 : Taking each of the character detected from previous step,
         and your features for each of the enrolled characters, you are required to a recognition or matching.

    Args:
        test_img : image that contains character to be detected.
        characters_list: list of characters along with name for each character.

    Returns:
    a nested list, where each element is a dictionary with {"bbox" : (x(int), y (int), w (int), h (int)), "name" : (string)},
        x: row that the character appears (starts from 0).
        y: column that the character appears (starts from 0).
        w: width of the detected character.
        h: height of the detected character.
        name: name of character provided or "UNKNOWN".
        Note : the order of detected characters should follow english text reading pattern, i.e.,
            list should start from top left, then move from left to right. After finishing the first line, go to the next line and continue.
        
    """
    # TODO Add your code here. Do not modify the return and input arguments
    # cv2.imshow("tet", test_img)
    # cv2.waitKey(0)  # wait for a keyboard input
    # cv2.destroyAllWindows()
    enrolled_images = enrollment(characters)

    cc_list = detection(test_img)

    recognition(enrolled_images, test_img, cc_list)

    # raise NotImplementedError


def enrollment(images_to_read):
    """ Args:
        You are free to decide the input arguments.
    Returns:
    You are free to decide the return.
    """
    enrolled_images = []
    # TODO: Step 1 : Your Enrollment code should go here.
    for image in images_to_read:
        # absolute_path = os.path.join(os.getcwd(), 'data', 'characters', image)
        # read_img = read_image(absolute_path, False)
        sift_object = cv2.SIFT_create(nfeatures=100)  # type: cv2.SIFT
        kp, descriptor = sift_object.detectAndCompute(image[1], None)  # type: list, list
        if descriptor.count() == 0:
            continue
        outfile = open(str(image[0]) + ".json", 'w')
        json.dump(np.asarray(descriptor).tolist(), outfile, indent=6)
        enrolled_images.append(outfile.name)

    return enrolled_images


def is_valid(vis, binary_image, row, col, max_row, max_col):
    if row < 0 or col < 0 or row >= max_row or col >= max_col:
        return False

    if vis[row][col]:
        return False

    # No point looking at empty pixels
    if binary_image[row][col] != 0:
        return False

    return True


def label_bfs(binary_image):
    max_row, max_col = np.asarray(binary_image).shape
    vis = [[False for _ in range(max_col)] for __ in range(max_row)]

    label = 1
    # Pairs as up, right, down, left
    d_row = [-1, 0, 1, 0]
    d_col = [0, 1, 0, -1]

    connected_components = []

    for row in range(0, max_row):
        for col in range(0, max_col):

            if not is_valid(vis, binary_image, row, col, max_row, max_col):
                continue

            q = [(row, col)]
            component_start_coord = (row, col)

            max_j = 0
            max_i = 0
            while len(q) > 0:
                cell = q.pop()
                i = cell[0]
                j = cell[1]
                if i > max_i:
                    max_i = i
                if j > max_j:
                    max_j = j

                binary_image[i][j] = label
                for k in range(4):
                    adj_i = i + d_row[k]
                    adj_j = j + d_col[k]
                    # print(adj_i, adj_j)
                    if is_valid(vis, binary_image, adj_i, adj_j, max_row, max_col):
                        # print("Updating connected components")
                        q.append((adj_i, adj_j))
                        vis[adj_i][adj_j] = True

            label += 1
            connected_components.append([component_start_coord, max_i, max_j])

    return binary_image, connected_components


def detection(test_img):
    """ 
    Use connected component labeling to detect various characters in an test_img.
    Args:
        You are free to decide the input arguments.
    Returns:
    You are free to decide the return.
    """

    # TODO: Step 2 : Your Detection code should go here.

    # Converting those pixels with values 1-127 to 255 and others to 255.
    _, binary_img = cv2.threshold(test_img, 100, 255, cv2.THRESH_BINARY)  # type: list, list

    # implement connected component with bfs

    labelled_image, connected_comps = label_bfs(deepcopy(binary_img))

    # with open("out.csv", "w", newline="") as f:
    #     writer = csv.writer(f)
    #     writer.writerows(labelled_image)

    return connected_comps


def recognition(enrolled_images, test_img, cc_list):
    """ 
    Args:
        You are free to decide the input arguments.
    Returns:
    You are free to decide the return.
    """
    # TODO: Step 3 : Your Recognition code should go here.

    # raise NotImplementedError


def save_results(coordinates, rs_directory):
    """
    Donot modify this code
    """
    results = []
    with open(os.path.join(rs_directory, 'results.json'), "w") as file:
        json.dump(results, file)


def main():
    args = parse_args()

    characters = []

    all_character_imgs = glob.glob(args.character_folder_path + "/*")

    for each_character in all_character_imgs:
        character_name = "{}".format(os.path.split(each_character)[-1].split('.')[0])
        characters.append([character_name, read_image(each_character, show=False)])

    test_img = read_image(args.test_img)

    results = ocr(test_img, characters)

    save_results(results, args.rs_directory)


if __name__ == "__main__":
    main()
