from readImage import *
import numpy as np
import random
import operator
import math
from copy import deepcopy
import csv
import time

loaded_images = list()

#number of attributes RBG -> 3
num_attr = 3 

#number of cluster 
k = 5

# Maximum number of iterations
MAX_ITER = 100

# Fuzzy parameter
m = 2.00
p = float(2/(m-1))

# Number of data points = so diem anh pixel
n = 256 * 256

# exponent
a = 0.5

# Threshold
thres = 0.005

# Results
durations = list()
membership_matrix_results = list()
cluster_center_results = list()
cluster_labels_results = list()

def random_membership_value():
    random_num_list = [random.random() for i in range(k)]
    summation = sum(random_num_list)
    random_n_list_list = [x/summation for x in random_num_list]
    return random_n_list_list

def check(list_):
    result = False
    result = all(elem >= 0 for elem in list_)
    return result

def check2(list_):
    result = False
    result = all(elem <= 1 for elem in list_)
    return result

def init_membership_values():
    #Each data point
    membership_values_list = np.array()
    random_e_list = list()
    random_n_list = list()

    random_u_list = [-1 for i in range(k)]
    # print(check2(random_n_list + random_u_list + random_e_list))
    # print(check(random_u_list))
    while (check(random_u_list) == False) | (check2(random_n_list + random_u_list + random_e_list) == False) :
        #random e 
        random_e_list = np.random.random(k)
        sum_e_list = sum(random_e_list)

        #calcu n
        sum_n_list = 1 - sum_e_list / k
        total = sum_n_list
        random_n_list = []
        for i in range(k-1):
            val = np.random.uniform(0,total)
            random_n_list.append(val)
            total -= val
        random_n_list.append(total)

        #calcu u
        membership_values = random_membership_value()
        random_u_list = np.divide(membership_values, (2 - random_e_list))
    
    #check constraints
    # print(random_e_list)
    # print(random_n_list)
    # print(random_u_list)
    for i in range(k):
        membership_values_list.append([random_u_list[i], random_n_list[i], random_e_list[i]])

    # print(membership_values_list)
    return membership_values_list

def checkConstraints(membership_matrix):
    sum1 = sum([membership_matrix[0][j][0] * (2 - membership_matrix[0][j][2]) for j in range(k)])
    print(sum1)
    sum2 = sum([membership_matrix[0][j][1] + membership_matrix[0][j][2] / k for j in range(k)])
    print(sum2)
    
def init_membership_matrix():
    membership_matrix = list()
    for i in range(n):
        membership_matrix.append(init_membership_values())
    # print(membership_matrix)
    print("init done")
    return membership_matrix


def calculateClusterCenter(membership_mat, data):
    cluster_mem_val = list(zip(*membership_mat))
    cluster_centers = list()
    numerator = 0
    denominator = 0
    start_time = time.time()
    for j in range(k):
        xraised =[((ele[0] * (2 - ele[2])) ** m) for ele in cluster_mem_val[j]] 
        denominator = sum(xraised)
        temp_num = list()
        for i in range(n):
            data_point = list(data[i])
            prod = [xraised[i] * val for val in data_point]
            temp_num.append(prod)
        numerator = map(sum, zip(*temp_num))
        center = [z/denominator for z in numerator]
        cluster_centers.append(center)
    end_time = time.time()
    runtime =  end_time - start_time
    print("Calculate cluster center excecute time: %.2fs" % runtime)
    return cluster_centers

def updateMembershipValue(membership_mat, cluster_centers):
    start_time = time.time()
    for i in range(n):
        data_point = list(data[i])
        distances = [distance(data_point, cluster_centers[j]) for j in range(k)]
        for j in range(k):
            # new u
            den = sum([(2 - membership_mat[i][c][2])*np.power(distances[j]/distances[c], p) for c in range(k)])
            membership_mat[i][j][0] = 1/den 
        
            # new n
            membership_mat[i][j][1] = np.exp(-membership_mat[i][j][2]) / sum([np.exp(-membership_mat[i][c][2]) for c in range(k)]) * (1 - sum([membership_mat[i][c][2] for c in range(k)])/5)
        
            # new e
            membership_mat[i][j][2] = 1 - (membership_mat[i][j][0] + membership_mat[i][j][1]) - np.power(1 - np.power((membership_mat[i][j][0] + membership_mat[i][j][1]), a), (1/a))

    end_time = time.time()
    runtime =  end_time - start_time
    print("Update membership excecute time: %.2fs" % runtime)

def distance(x, y):
    return np.linalg.norm(list(map(operator.sub, x, y)))

# def matrix_distance(matrix1, matrix2):
#     dis = 0.0
#     sum_u = 0.0
#     sum_n = 0.0
#     sum_e = 0.0
#     start_time = time.time()
#     for i in range(n):
#         for j in range(k):
#             sum_u += np.square(matrix1[i][j][0] - matrix2[i][j][0])
#             sum_n += np.square(matrix1[i][j][1] - matrix2[i][j][1])
#             sum_e += np.square(matrix1[i][j][2] - matrix2[i][j][2])
    
#     dis = np.sqrt(sum_u) + np.sqrt(sum_e) + np.sqrt(sum_n)
#     end_time = time.time()
#     runtime =  end_time - start_time
#     print("Calculate distance excecute time: %.2fs" % runtime)
#     return dis

def matrix_distance(matrix1, matrix2):
    dis = 0.0
    start_time = time.time()
    list_u1 = list()
    list_n1 = list()
    list_e1 = list()
    list_u2 = list()
    list_n2 = list()
    list_e2 = list()
    for i in range(n):
        list_u1 += list(map(list, zip(*matrix1[i])))[0]
        list_n1 += list(map(list, zip(*matrix1[i])))[1]
        list_e1 += list(map(list, zip(*matrix1[i])))[2]
        list_u2 += list(map(list, zip(*matrix2[i])))[0]
        list_n2 += list(map(list, zip(*matrix2[i])))[1]
        list_e2 += list(map(list, zip(*matrix2[i])))[2]
        
    dis = distance(list_e1,list_e2) + distance(list_n1,list_n2) + distance(list_u1,list_u2)
    end_time = time.time()
    runtime =  end_time - start_time
    print("Calculate distance excecute time: %.2fs" % runtime)
    return dis

def PFS(data):
    membership_matrix = init_membership_matrix()
    curr = 0
    while curr <= MAX_ITER:
        print("Current iter:", curr)
        membership_matrix_old = deepcopy(membership_matrix)
        cluster_center = calculateClusterCenter(membership_matrix, data)
        updateMembershipValue(membership_matrix, cluster_center)
        dis = matrix_distance(membership_matrix_old, membership_matrix)
        print("Current distance:", dis)
        if dis <= thres:
            print("Iter done!")
            break
        curr += 1
    cluster_labels = getClusters(membership_matrix)
    return membership_matrix, cluster_center, cluster_labels


def getClusters(membership_mat):
    cluster_labels = list()
    for i in range(n):
        max_val, idx = max((val[0], idx) for (idx, val) in enumerate(membership_mat[i]))
        cluster_labels.append(idx)
    # print(cluster_labels)
    return cluster_labels

def fuzzyClusterImage(cluster_labels, cluster_center):
    result_data = np.zeros(shape=(n, 3))
    for i in range(n):
        result_data[i] = cluster_center[cluster_labels[i]]
    return result_data

def writeMatrixToCsv(filename, matrix):
    with open('results/' + filename, "w+") as f:
        csv_write = csv.writer(f, delimiter = ',')
        for i in range(len(matrix)):
            csv_write.writerows(matrix[i])

def writeToCsv(filename, list_):
    with open('results/' + filename, "w+") as f:
        csv_write = csv.writer(f, delimiter = ',')
        csv_write.writerows(list_)

if __name__ == "__main__":
    loaded_images, image_names = loadImageFromFile('images')
    for (index, loaded_image) in enumerate(loaded_images):
        start_time = time.time()
        print("Processing image %d %s" % (index, image_names[index]))
        data = to2dArray(loaded_image)
        # data = normalizeData(data)
        membership_matrix_result, cluster_center_result, cluster_labels_result = PFS(data)
        result_data = fuzzyClusterImage(cluster_labels_result, cluster_center_result)
        image = from2dArrayToImage(result_data)
        image_name = image_names[index][:-4]
        image.save(image_name + "_after" + ".jpg")
        end_time = time.time()
        duration =  end_time - start_time

        writeMatrixToCsv(image_name + '_matrix' + '.csv', membership_matrix_result)
        writeToCsv(image_name + '_cluster_center' + '.csv', cluster_center_result)
        # durations.append(duration)
        # membership_matrix_results.append(membership_matrix_result)
        # cluster_center_results.append(cluster_center_result)
        # cluster_labels_results.append(cluster_labels_result)
        print("Excecute time image %s: %.6fs" % (image_names[index], duration))
    
    # return membership_matrix_results, cluster_center_results, cluster_labels_results, durations
    
    

    