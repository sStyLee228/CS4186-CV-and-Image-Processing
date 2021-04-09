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


def sift_flann(im_query, im, i):
    sift = cv.SIFT_create()
    kp1, des1 = sift.detectAndCompute(im_query, None)
    kp2, des2 = kp_desc_dict[i]

    index_params = dict(algorithm = 1, trees = 5)
    search_params = dict(checks = 50)

    flann = cv.FlannBasedMatcher(index_params, search_params)
    matches = flann.knnMatch(des1, des2, k = 2)
    good_matches = [m for m, n in matches if m.distance < 0.89 * n.distance]

    print('There are %d good matches' % len(good_matches))
    similarity = (len(good_matches) * 100) / len(kp2)
    print('Similarity is %.2f percent' % similarity)

    if len(good_matches) > 5:
        src_pts = np.float32([kp1[m.queryIdx].pt for m in good_matches]).reshape(-1, 1, 2)
        dst_pts = np.float32([kp2[m.trainIdx].pt for m in good_matches]).reshape(-1, 1, 2)

        M, mask = cv.findHomography(src_pts, dst_pts, cv.RANSAC, 5.0)
        matchesMask = mask.ravel().tolist()
        h, w = im_query.shape
        pts = np.float32([[0, 0], [0, h-1], [w-1, h-1], [w-1, 0]]).reshape(-1, 1, 2)
        dst = cv.perspectiveTransform(pts, M)

        im = cv.polylines(im, [np.int32(dst)], True, 255, 3, cv.LINE_AA)

    else:
        print ("Not enough matches are found - {}/{}".format(len(good), 10))
        matchesMask = None

    if matchesMask is None:
        print('trash')
    else:
        inlier_matches = sum(matchesMask)

    draw_params = dict(matchColor = (0,255,0), # draw matches in green color
                       singlePointColor = None,
                       matchesMask = matchesMask, # draw only inliers
                       flags = cv.DrawMatchesFlags_DEFAULT)

    im_final = cv.drawMatches(im_query, kp1, im, kp2, good_matches, None, **draw_params)

    return inlier_matches

if __name__ == '__main__':
    # NOTE THAT INDEX STARTS FROM ZERO WHILE IMAGE NUMBERS START FROM 1

    images = [cv.imread('Images\\{}.jpg'.format(str(file).zfill(4)), 0) for file in range(1, 5001)]
    given_boundary_boxes = [open('Images\\{}'.format(file)).readline().rstrip('\n').split() for file in os.listdir('Images') if file.endswith('.txt')]
    given_boundary_boxes = [[int(x) for x in row] for row in given_boundary_boxes]
    images_cropped = [image[box[1]:box[1]+box[3], box[0]:box[0]+box[2]] for image, box in zip(images, given_boundary_boxes)] + [image for image in images[2000:]]

    query_images = [cv.imread('Queries\\{}'.format(file), 0) for file in os.listdir('Queries') if file.endswith('.jpg')]
    query_boundary_boxes = [open('Queries\\{}'.format(file)).readline().rstrip('\n').split() for file in os.listdir('Queries') if file.endswith('.txt')]
    query_boundary_boxes = [[int(x) for x in row] for row in query_boundary_boxes]
    query_images_cropped = [image[box[1]:box[1]+box[3], box[0]:box[0]+box[2]] for image, box in zip(query_images, query_boundary_boxes)]

    example_query_images = [cv.imread('examples\\example_query\\{}'.format(file), 0) for file in os.listdir('examples\\example_query') if file.endswith('.jpg')]
    example_query_boundary_boxes = [open('examples\\example_query\\{}'.format(file)).readline().rstrip('\n').split() for file in os.listdir('examples\\example_query') if file.endswith('.txt')]
    example_query_boundary_boxes = [[int(x) for x in row] for row in example_query_boundary_boxes]
    example_query_images_cropped = [image[box[1]:box[1]+box[3], box[0]:box[0]+box[2]] for image, box in zip(example_query_images, example_query_boundary_boxes)]

    sift = cv.SIFT_create()
    kp_desc_dict = []
    for i, image in enumerate(images_cropped):
        print('Calculating descriptors for image #%d' % (i + 1))
        kp_desc_dict.append(sift.detectAndCompute(image, None))
    # kp_desc_dict = [sift.detectAndCompute(image, None) for image in images_cropped]

    # GET RANKLIST FOR 20 QUERY IMAGES
    # rank_list = open('rankList.txt', mode = 'a')
    # for i, query_image in enumerate(query_images_cropped):
    #     print('Query #%d starts' % (i + 1))
    #
    #     single_rank_list = {str(j + 1): sift_flann(query_image, image, j) for j, image in enumerate(images_cropped)}
    #     print ('single_rank_list # %d is done' % (i + 1))
    #
    #     line = 'Q{}: '.format(i + 1) + ' '.join(str(key) for key in sorted(single_rank_list.items(), key = lambda x: x[1], reverse = True).keys())
    #     print('Query #%d is done' % (i + 1))
    #     rank_list.write(line)
    #
    # rank_list.close()

    # GET RANKLIST FOR 10 EXAMPLE IMAGES (FOR PERFORMANCE EVALUATION)
    rank_list = open('rankList.txt', mode = 'a')
    for i, example_image in enumerate(example_query_images_cropped):
        print('Example image #%d starts' % (i + 1))

        single_rank_list = {str(j + 1): sift_flann(example_image, image, j) for j, image in enumerate(images_cropped)}
        print ('single_rank_list # %d is done' % (i + 1))

        line = 'Q{}: '.format(i + 1) + ' '.join(str(key) for key in sorted(single_rank_list.items(), key = lambda x: x[1], reverse = True))
        print('Example image #%d is done' % (i + 1))
        rank_list.write(line)

    rank_list.close()
