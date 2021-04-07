import os
import sys
import numpy as np
import pandas as pd
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
from skimage import color, io
import scipy
from PIL import Image, ImageDraw

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

images = {}
for i in range(1, 5001):
    index = str(i).zfill(4)
    images[index] = Image.open('Images\{}.jpg'.format(index))

ground_truth_dict = readfile_as_dict('examples\example_result\rank_groundtruth.txt')
ranklist_example_dict = readfile_as_dict('examples\example_result\rankList.txt')

print(ground_truth_dict)
print(ranklist_example_dict)
