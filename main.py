from readImage import *
import numpy as np
import random
import operator
import math

loaded_images = list()

#number of attributes RBG -> 3
num_attr = 3 

#number of cluster 
k = 5

# Maximum number of iterations
MAX_ITER = 100

# Fuzzy parameter
m = 2.00

# Number of data points = so diem anh pixel
n = 0

def initializeMembershipMatrix():
    membership_matrix = list()
    for i in range(n):
        random_num_list = [random.random() for i in range(k)]
        summation = sum(random_num_list)
        temp_list = [x/summation for x in random_num_list]
        membership_matrix.append(temp_list)
    print(len(membership_matrix))
    return membership_matrix
    
if __name__ == "__main__":
    loaded_images = loadImageFromFile('images')
    test_image = loaded_images[0]
    shape = test_image.shape
    n = shape[0] * shape[1]
    initializeMembershipMatrix()
    # readPixel(test_image)
    # for loaded_image in loaded_images:
    #     print(loaded_image.shape)