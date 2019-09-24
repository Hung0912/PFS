from readImage import *
import numpy as np
import random
import operator
import math
from copy import deepcopy
import csv
import time

#number of cluster 
k = 10

# Maximum number of iterations
MAX_ITER = 1000

# Fuzzy parameter
m = 2.00
p = float(2/(m-1))

# Number of data points = so diem anh pixel
n = 256 * 256

# exponent
alpha = 0.5

# Threshold
thres = 0.001

file_path = 'results/'

def random_with_sum(suma):
    a = np.random.rand(k,1)
    summation = np.sum(a, axis= 0)
    a = (a / summation) * suma
    return a

def init_membership_values():
    #Each data point
    while 1 :
        #random e 
        e_arr = np.random.rand(k,1)
        sum_e_arr = np.sum(e_arr, axis= 0)

        #calcu n
        sum_n_arr = 1 - sum_e_arr / k
        n_arr = random_with_sum(sum_n_arr)

        #calcu u
        rando = random_with_sum(1)
        u_arr = rando / (2 - e_arr)
    
        s = u_arr + e_arr + n_arr

        if (np.all(s<=1, axis= 0) & np.all(e_arr>=0, axis= 0) & np.all(n_arr>=0, axis= 0) & np.all(u_arr>=0, axis= 0)):
            break  
    result_matrix = np.concatenate((u_arr, n_arr, e_arr), axis= 1)
    return result_matrix

    
def init_membership_matrix():
    membership_matrix = np.zeros((n,5,3))
    for i in range(n):
        membership_matrix[i] = init_membership_values()
    # # print(membership_matrix)
    print("init done")
    return membership_matrix

def norm(x, y):
    return np.linalg.norm(x-y)

def distance_matrix(matrix1, matrix2):
    matrix1 = matrix1.reshape(n*k,3)
    u_matrix1 = matrix1[:,0]
    n_matrix1 = matrix1[:,1]
    e_matrix1 = matrix1[:,2]
    matrix2 = matrix2.reshape(n*k,3)
    u_matrix2 = matrix2[:,0]
    n_matrix2 = matrix2[:,1]
    e_matrix2 = matrix2[:,2]
    return norm(u_matrix1,u_matrix2) + norm(e_matrix1,e_matrix2) + norm(n_matrix1,n_matrix2)


def calculateClusterCenter(matrix, data):
    u_matrix = matrix[:,:,0]
    e_matrix = matrix[:,:,2]

    a = ((u_matrix*(2-e_matrix)) ** m).reshape(n,5,1)
    b = a * data.reshape(n,1,3)
    tuso = np.sum(b,axis = 0)
    mauso = np.sum(a,axis = 0)
    result = tuso / mauso
    return result

def updateMembershipMatrix(matrix, cluster_centers):
    u_matrix = matrix[:,:,0]
    n_matrix = matrix[:,:,1]
    e_matrix = matrix[:,:,2]

    #cal new n matrix
    a = np.exp(-e_matrix)
    b = np.sum(np.exp(-e_matrix),axis= 1)
    c = 1 - np.sum(e_matrix,axis= 1) / 5
    n_matrix = a * (c / b).reshape(n,1)
    
    x = np.repeat(data, k, axis= 0)
    v = np.vstack([cluster_centers] * n)
    a = np.linalg.norm(x-v, axis= 1).reshape((n,k))
    b = (2 - e_matrix) * (a ** p) * np.sum(1/(a ** p), axis= 1).reshape(n,1)
    u_matrix = 1 / b


    #cal new e matrix
    e_matrix = 1 - (u_matrix + n_matrix) - (1-(u_matrix + n_matrix)**alpha)**(1/alpha)

    result_matrix = np.zeros((n,k,3))
    result_matrix[:,:,0] = u_matrix
    result_matrix[:,:,1] = n_matrix
    result_matrix[:,:,2] = e_matrix
    return result_matrix

def afterClusterData(data, membership_mat, cluster_centers):
    result_data = np.zeros(shape=(n, 3))
    positive = membership_mat[:,:,0]
    for i in range(n):
        a = positive[i]
        result = np.where(a == np.amax(a))
        index = result[0][0]
        result_data[i] = cluster_centers[index]
    return result_data

def PFS(data):
    membership_matrix = init_membership_matrix()
    curr = 0
    while curr <= MAX_ITER:
        tic = time.time()
        print("Current iter:", curr)
        membership_matrix_old = deepcopy(membership_matrix)
        cluster_centers = calculateClusterCenter(membership_matrix, data)
        membership_matrix = updateMembershipMatrix(membership_matrix, cluster_centers)
        dis = distance_matrix(membership_matrix_old, membership_matrix)
        print("Current distance:", dis)
        if dis <= thres:
            print("Iter done!")
            break
        curr += 1
        toc = time.time()
        print("time: %0.2fs" %(toc-tic))
    return membership_matrix, cluster_centers

def save_membership_matrixs(filename, matrix):
    with open(file_path + 'membership_matrixs/'  + filename, "w+") as f:
        csv_write = csv.writer(f, delimiter = ',')
        for i in range(matrix.shape[0]):
            csv_write.writerows(matrix[i])

def save_cluster_centers(filename, matrix):
    with open(file_path + 'cluster_centers/' + filename, "w+") as f:
        csv_write = csv.writer(f, delimiter = ',')
        csv_write.writerows(matrix)

if __name__ == "__main__":
    loaded_images, image_names = loadImageFromFile('images')
    for (index, loaded_image) in enumerate(loaded_images):
        data = loaded_image
        start_time = time.time()
        print("Processing image %d %s" % (index, image_names[index]))
        # data = normalizeData(data)
        result_membership_matrix, result_cluster_centers = PFS(data)
        after_data = afterClusterData(data, result_membership_matrix, result_cluster_centers)
        image = vector2Image(after_data)
        image_name = image_names[index][:-4]
        image.save( file_path + 'images/' + image_name + "_after" + ".jpg")
        end_time = time.time()
        duration =  end_time - start_time
        print("Excecute time image %s: %.6fs" % (image_names[index], duration))

        save_membership_matrixs(image_name + '_matrix' + '.csv', result_membership_matrix)
        save_cluster_centers(image_name + '_cluster_center' + '.csv', result_cluster_centers)
        