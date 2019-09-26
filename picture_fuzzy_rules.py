import numpy as np
import random
import operator
import math
from picture_fuzzy_clustering import *
from cropper import *

datas, matrixs = cropper()

# for i in range(len(datas)):
if 0 == 0:
    data = datas[0]
    matrix = matrixs[0]
    data = data.reshape((data.shape[0] * data.shape[1], data.shape[2]))
    matrix = matrix.reshape((matrix.shape[0] * matrix.shape[1], matrix.shape[2], matrix.shape[3]))
    # print(data.shape)
    # print(matrix.shape)
    n1 = matrix.shape[0]
    k1 = matrix.shape[1]

    # cal new cluster_center
    new_cluster_center = calculateClusterCenter(matrix, data, n1, k1)

    defuzz = np.zeros((k1,3))
    # cal triangular picture fuzzy number of fuzzy rules
    for j in range(k1):
        u_matrix = matrix[:,j,0]
        n_matrix = matrix[:,j,1]
        e_matrix = matrix[:,j,2]
        u = (u_matrix + n_matrix) / (1 + e_matrix)

        a_ = np.amin(data, axis= 0)
        c_ = np.amax(data, axis= 0)
        b = new_cluster_center[j]
        a = np.zeros(3)
        c = np.zeros(3)
        for k in range(3):
            color = data[:,k]
            #a
            result = np.where(data <= b[k] )
            indexs = result[0]
            a[k] = np.sum(u[indexs] * color[indexs]) / np.sum(u[indexs])
            #c
            result = np.where(data >= b[k] )
            indexs = result[0]
            c[k] = np.sum(u[indexs] * color[indexs]) / np.sum(u[indexs])
        defuzz[j] = (1*a_ + 2*a + 3*b + 2*c + 1*c_) / (1+2+3+2+1)

    print(defuzz)