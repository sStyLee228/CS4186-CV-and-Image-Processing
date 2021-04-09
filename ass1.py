import os
import sys
import numpy as np
import pandas as pd
from matplotlib import image as mpimg
from matplotlib import pyplot as plt
from skimage import color, io, feature, viewer
import scipy
from scipy.spatial import distance as sp_distance
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
    good_matches = [m for m, n in matches if m.distance < 0.7 * n.distance]

    print('Image # %d' % (i+1))

    if len(good_matches) >= 5:
        src_pts = np.float32([kp1[m.queryIdx].pt for m in good_matches]).reshape(-1, 1, 2)
        dst_pts = np.float32([kp2[m.trainIdx].pt for m in good_matches]).reshape(-1, 1, 2)

        M, mask = cv.findHomography(src_pts, dst_pts, cv.RANSAC, 5.0)
        matchesMask = mask.ravel().tolist()

    else:
        print("Not enough matches are found - {}/{}".format(len(good_matches), 5))
        matchesMask = None

    if matchesMask is None:
        inlier_matches = 0
    else:
        inlier_matches = sum(matchesMask)

    return inlier_matches

def create_rgb_hist(image):
    h, w, c = image.shape
    rgb_hist = np.zeros([16 * 16 * 16, 1], np.float32)
    bsize = 256 / 16
    for row in range(h):
        for col in range(w):
            b = image[row, col, 0]
            g = image[row, col, 1]
            r = image[row, col, 2]
            index = int(b / bsize) * 16 * 16 + int(g / bsize) * 16 + int(r / bsize)
            rgb_hist[int(index), 0] += 1
    return rgb_hist

def compareHistogram(query, image):
    query_hist, image_hist = create_rgb_hist(image), create_rgb_hist(query)
    dist = cv.compareHist(query_hist, image_hist, cv.HISTCMP_CORREL)
    return dist

def showImagesHorizontally(list_of_files):
    fig = plt.figure()
    number_of_files = len(list_of_files)
    for i in range(number_of_files):
        a=fig.add_subplot(1,number_of_files,i+1)
        image = mpimg.imread('Images\\{}'.format(list_of_files[i]))
        plt.imshow(image,cmap='Greys_r')
        plt.axis('off')
    plt.show()

if __name__ == '__main__':
    # NOTE THAT INDEX STARTS FROM ZERO WHILE IMAGE NUMBERS START FROM 1
    images = [cv.imread('Images\\{}.jpg'.format(str(file).zfill(4))) for file in range(1, 5001)]
    given_boundary_boxes = [open('Images\\{}'.format(file)).readline().rstrip('\n').split() for file in os.listdir('Images') if file.endswith('.txt')]
    given_boundary_boxes = [[int(x) for x in row] for row in given_boundary_boxes]
    images_cropped = [image[box[1]:box[1]+box[3], box[0]:box[0]+box[2]] for image, box in zip(images, given_boundary_boxes)] + [image for image in images[2000:]]

    query_images = [cv.imread('Queries\\{}'.format(file)) for file in os.listdir('Queries') if file.endswith('.jpg')]
    query_boundary_boxes = [open('Queries\\{}'.format(file)).readline().rstrip('\n').split() for file in os.listdir('Queries') if file.endswith('.txt')]
    query_boundary_boxes = [[int(x) for x in row] for row in query_boundary_boxes]
    query_images_cropped = [image[box[1]:box[1]+box[3], box[0]:box[0]+box[2]] for image, box in zip(query_images, query_boundary_boxes)]

    example_query_images = [cv.imread('examples\\example_query\\{}'.format(file)) for file in os.listdir('examples\\example_query') if file.endswith('.jpg')]
    example_query_boundary_boxes = [open('examples\\example_query\\{}'.format(file)).readline().rstrip('\n').split() for file in os.listdir('examples\\example_query') if file.endswith('.txt')]
    example_query_boundary_boxes = [[int(x) for x in row] for row in example_query_boundary_boxes]
    example_query_images_cropped = [image[box[1]:box[1]+box[3], box[0]:box[0]+box[2]] for image, box in zip(example_query_images, example_query_boundary_boxes)]

    # FOR COLOR HISTOGRAM METHOD (QUITE SLOW)
    # print(np.array(example_query_images_cropped[0]).shape)
    # example_images_hists = []
    # for i, image in enumerate(example_query_images_cropped):
    #     example_images_hists.append(create_rgb_hist(np.array(image)))
    #     print('Histogram for Example Image %d' % (i+1))
    #
    # images_hists = []
    # for i, image in enumerate(images_cropped):
    #     images_hists.append(create_rgb_hist(np.array(image)))
    #     print('Histogram for Image %d' % (i+1))
    #
    # rank_list = open('rankListHist.txt', mode = 'a')
    # for i, example_image_hist in enumerate(example_images_hists):
    #     print('Example image #%d starts' % (i + 1))
    #
    #     single_rank_list = {str(j + 1): compareHistogram(example_image_hist, image_hist) for j, image in enumerate(images_hists)}
    #     print ('single_rank_list # %d is done' % (i + 1))
    #
    #     line = 'Q{}: '.format(i + 1) + ' '.join(tuple[0] for tuple in sorted(single_rank_list.items(), key = lambda x: x[1], reverse = True))
    #     print('Example image #%d is done' % (i + 1))
    #     rank_list.write(line + '\n')
    #
    # rank_list.close()

    # SAVE THE DESCRIPTORS AND KEYPOINTS AND USE THEM TO EFFICIENTLY COMPUTE THE SIMILARITIES
    sift = cv.SIFT_create()
    kp_desc_dict = []
    for i, image in enumerate(images_cropped):
        print('Calculating descriptors for image #%d' % (i + 1))
        kp_desc_dict.append(sift.detectAndCompute(image, None))

    # GET RANKLIST FOR 20 QUERY IMAGES (FOR SUBMISSION)
    rank_list = open('rankList.txt', mode = 'a')
    for i, query_image in enumerate(query_images_cropped):
        print('Query #%d starts' % (i + 1))

        single_rank_list = {str(j + 1): sift_flann(query_image, image, j) for j, image in enumerate(images_cropped)}
        print ('single_rank_list # %d is done' % (i + 1))

        line = 'Q{}: '.format(i + 1) + ' '.join(tuple[0] for tuple in sorted(single_rank_list.items(), key = lambda x: x[1], reverse = True))
        print('Query #%d is done' % (i + 1))
        rank_list.write(line)

    rank_list.close()

    # GET RANKLIST FOR 10 EXAMPLE IMAGES (FOR PERFORMANCE EVALUATION)
    # rank_list = open('rankList.txt', mode = 'a')
    # for i, example_image in enumerate(example_query_images_cropped):
    #     print('Example image #%d starts' % (i + 1))
    #
    #     single_rank_list = {str(j + 1): sift_flann(example_image, image, j) for j, image in enumerate(images_cropped)}
    #     print ('single_rank_list # %d is done' % (i + 1))
    #
    #     line = 'Q{}: '.format(i + 1) + ' '.join(tuple[0] for tuple in sorted(single_rank_list.items(), key = lambda x: x[1], reverse = True))
    #     print('Example image #%d is done' % (i + 1))
    #     rank_list.write(line + '\n')
    #
    # rank_list.close()

    # FOR RETRIEVAL OF TOP 10 IMAGES BY QUERY FOR REPORT PURPOSES
    # sift_flann_toshow_list = list(map(lambda file: str(file).zfill(4) + '.jpg', readfile_as_dict('rankList_07_inline_equals_five.txt')['Q5:'][:10]))
    #
    # print(sift_flann_toshow_list)
    # showImagesHorizontally(sift_flann_toshow_list)

    # plt.imshow(example_query_images_cropped[4])
    # plt.show()
