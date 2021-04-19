# CS4186-CV-and-Image-Processing

This repo is intended to demonstrate the experience I had throughout CS4186 and my completed assignment for skills demonstration purpose.

# Objectives

This assignment requires to **implement instance search** on the given 20 query images (see folder [Queries](https://github.com/sStyLee228/CS4186-CV-and-Image-Processing/tree/main/Queries)) **using the given 5000 images** and **rank them according to the similarity rate**. The performance is evaluated by **Mean Average Precision (MAP)** (see [examples/example_result/metric_map.py](https://github.com/sStyLee228/CS4186-CV-and-Image-Processing/blob/main/examples/example_result/metric_map.py)) 

The assignment also requires to implement **2 different** algorithms for instance search (e.g. Color Histogram, SIFT, CNN, etc.). I have implemented SIFT and Color Histogram, though SIFT significantly outperforms with around 34% precision (around 1 hour execution time), compared to around 3% for Color Histogram (around 5 hours execution time).
