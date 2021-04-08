import os
import sys
import numpy as np
import pandas as pd
from matplotlib import image as mpimg
from matplotlib import pyplot as plt
from skimage import color, io, feature, viewer
import scipy
from PIL import Image, ImageDraw, ImageFilter
import cv2 as cv

this_file_path = sys.argv[0]

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

def calculateDistance(im1, im2):
    return np.sum((im1-im2)**2)

def calcEuclidean(im1, im2):
    df = np.asarray(im1 - im2)
    return np.sqrt(np.sum(df**2, axis=1))

ground_truth_dict = readfile_as_dict('examples\\example_result\\rank_groundtruth.txt')
ranklist_example_dict = readfile_as_dict('examples\\example_result\\rankList.txt')

# NOTE THAT INDEX STARTS FROM ZERO WHILE IMAGE NUMBERS START FROM 1
images_range = 1001

images = [cv.imread('Images\\{}.jpg'.format(str(file).zfill(4)), 0) for file in range(1, images_range)]
given_boundary_boxes = [open('Images\\{}'.format(file)).readline().rstrip('\n').split() for file in os.listdir('Images') if file.endswith('.txt')]
given_boundary_boxes = [[int(x) for x in row] for row in given_boundary_boxes[:images_range]]
images_cropped = [image[box[1]:box[1]+box[3], box[0]:box[0]+box[2]] for image, box in zip(images, given_boundary_boxes)]

query_images = [cv.imread('Queries\\{}'.format(file), 0) for file in os.listdir('Queries') if file.endswith('.jpg')]
query_boundary_boxes = [open('Queries\\{}'.format(file)).readline().rstrip('\n').split() for file in os.listdir('Queries') if file.endswith('.txt')]
query_boundary_boxes = [[int(x) for x in row] for row in query_boundary_boxes]
query_images_cropped = [image[box[1]:box[1]+box[3], box[0]:box[0]+box[2]] for image, box in zip(query_images, query_boundary_boxes)]

im_query = query_images_cropped[0]
im = images_cropped[252]

orb = cv.ORB_create()
sift = cv.SIFT_create()
bf = cv.BFMatcher()

kp1, des1 = sift.detectAndCompute(im_query, None)
kp2, des2 = sift.detectAndCompute(im, None)

index_params = dict(algorithm = 0, trees = 5)
search_params = dict(checks = 50)

# flann = cv.FlannBasedMatcher(index_params, search_params)
# matches = flann.knnMatch(des1, des2, k = 2)

matches = bf.knnMatch(des1, des2, k = 2)
good = [[m] for m, n in matches if m.distance < 0.75 * n.distance]
matches_mask = [[1, 0] if m.distance < 0.75 * n.distance else [0, 0] for (m, n) in matches]
print('There are %d good matches' % len(good))
print('Similarity is %.2f percent' % ((len(good) * 100) / len(kp2)))
print(len(good), len(kp1), len(kp2))

# if len(good) > 10:
#     src_pts = np.float32([kp1[m.queryIdx].pt for m in good]).reshape(-1, 1, 2)
#     dst_pts = np.float32([kp2[m.trainIdx].pt for m in good]).reshape(-1, 1, 2)
#
#     M, mask = cv.findHomography(src_pts, dst_pts, cv.RANSAC, 5.0)
#     matchesMask = mask.ravel().tolist()
#
#     h, w = im_query.shape
#     pts = np.float32([[0, 0], [0, h-1], [w-1, h-1], [w-1, 0]]).reshape(-1, 1, 2)
#     dst = cv.perspectiveTransform(pts, M)
#
#     im = cv.polylines(im, [np.int32(dst)], True, 255, 3, cv.LINE_AA)
#
# else:
#     print ("Not enough matches are found - {}/{}".format(len(good), 10))
#     matchesMask = None
#
# draw_params = dict(matchColor = (0,255,0), # draw matches in green color
#                    singlePointColor = None,
#                    matchesMask = matches_mask, # draw only inliers
#                    flags = 2)
#
# im_final = cv.drawMatches(im_query, kp1, im, kp2, good, None, **draw_params)

im_final = cv.drawMatchesKnn(im_query, kp1, im, kp2, good, None, flags = 2)

plt.imshow(im_final)
plt.show()
