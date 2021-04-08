import os
import sys
import numpy as np
import pandas as pd
from matplotlib import image as mpimg
from matplotlib import pyplot as plt
from skimage import color, io
import scipy
from PIL import Image, ImageDraw
import cv2 as cv

def readfile_as_dict(path):
    with open(path) as f:
        file = f.readlines()
    ans_list = [line.strip() for line in file]
    ans_dict = {}
    for line in ans_list:
        for i, char in enumerate(line):
            if char == ':':
                key = line[:i+1]
                value = [int(num) for num in line[i+2:].split() if num.isdigit()]
                ans_dict[key] = value
    return ans_dict

ground_truth_dict = readfile_as_dict('examples\\example_result\\rank_groundtruth.txt')
ranklist_example_dict = readfile_as_dict('examples\\example_result\\rankList.txt')
given_boundary_boxes = [list(open('Images\\{}'.format(file)).readline().rstrip('\n').split()) for file in os.listdir('Images') if file.endswith('.txt')]
images = {str(i).zfill(4): Image.open('Images\\{}.jpg'.format(str(i).zfill(4))) for i in range(1, 5001)}

print(images)
print(given_boundary_boxes)
print(len(given_boundary_boxes))
