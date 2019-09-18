from readImage import *
import numpy as np
import random
import operator
import math
from copy import deepcopy

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

# exponent
a = 0.8

# Threshold
thres = 3000

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
    membership_values_list = list()
    random_e_list = list()
    random_n_list = list()

    random_u_list = [-1 for i in range(k)]
    while check(random_u_list) == False & check2(random_n_list + random_u_list + random_e_list) == False:
        #random e 
        random_e_list = np.random.random(5)
        sum_e_list = sum(random_e_list)

        #calcu n
        sum_n_list = 1 - sum_e_list / 5
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

    return cluster_centers

def updateMembershipValue(membership_mat, cluster_centers):
    p = float(2/(m-1))
    for i in range(n):
        data_point = list(data[i])
        distances = [matrix_distance(data_point, cluster_centers[j]) for j in range(k)]
        for j in range(k):
            # new u
            den = sum([(2 - membership_mat[i][j][2])*math.pow(float(distances[j]/distances[c]), p) for c in range(k)])
            membership_mat[i][j][0] = float(1/den)   
        
            # new n
            membership_mat[i][j][1] = np.exp(-membership_mat[i][j][2]) / sum([np.exp(-membership_mat[i][c][2]) for c in range(k)]) * (1 - sum([membership_mat[i][c][2] for c in range(k)])/5)
        
            # new e
            membership_mat[i][j][2] = 1 - (membership_mat[i][j][0] + membership_mat[i][j][1]) - np.power(1 - np.power((membership_mat[i][j][0] + membership_mat[i][j][1]), a), (1/a))

def matrix_distance(x, y):
    return np.linalg.norm(list(map(operator.sub, x, y)))

def checkThreshold(matrix1, matrix2):
    dis = 0.0
    for i in range(n):
        for j in range(k):
            dis += sum([np.absolute(matrix1[i][j][g] - matrix2[i][j][g]) for g in range(3)])
    return dis

def PFS(data):
    membership_matrix = init_membership_matrix()
    print(membership_matrix[0][2])
    curr = 0
    while curr <= MAX_ITER:
        membership_matrix_old = deepcopy(membership_matrix)
        cluster_center = calculateClusterCenter(membership_matrix, data)
        updateMembershipValue(membership_matrix, cluster_center)
        print(curr)
        print(checkThreshold(membership_matrix_old, membership_matrix))
        if checkThreshold(membership_matrix_old, membership_matrix) <= thres:
            print("check done!")
            break
        curr += 1
    return membership_matrix, cluster_center

if __name__ == "__main__":

    loaded_images = loadImageFromFile('images')
    # readPixel(test_image)

    # for loaded_image in loaded_images:
    #     print(loaded_image.shape)

    test_image = loaded_images[0]
    # print(test_image.shape)

    data = readPixel(test_image)
    n = test_image.shape[0] * test_image.shape[1]
    
    membership_matrix_result, cluster_center_result = PFS(data)

    print(membership_matrix_result[0][0])
    print(cluster_center_result)
    