import numpy as np
import random
import operator
import math
from picture_fuzzy_clustering import *
from cropper import *
from readImage import readImage

datas, matrixs = cropper()
data2Image(datas[0]).show()

A = np.zeros((len(datas),k,3))
C = np.zeros((len(datas),k,3))
B = np.zeros((len(datas),k,3))
A_ = np.zeros((len(datas),k,3))
C_ = np.zeros((len(datas),k,3))

for i in range(len(datas)):
    data = datas[i]
    matrix = matrixs[i]
    data = data.reshape((data.shape[0] * data.shape[1], data.shape[2]))
    matrix = matrix.reshape((matrix.shape[0] * matrix.shape[1], matrix.shape[2], matrix.shape[3]))
    # print(data.shape)
    # print(matrix.shape)
    n1 = matrix.shape[0]
    k1 = matrix.shape[1]

    # cal new cluster_center
    new_cluster_center = calculateClusterCenter(matrix, data, n1, k1)

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
            result = np.where(color <= b[k] )
            indexs = result[0]
            a[k] = np.sum(u[indexs] * color[indexs]) / np.sum(u[indexs])
            if a[k] < a_[k]:
                a[k] = a_[k]
            #c
            result = np.where(color >= b[k] )
            indexs = result[0]
            c[k] = np.sum(u[indexs] * color[indexs]) / np.sum(u[indexs])
            if c[k] > c_[k]:
                c[k] = c_[k]
        A_[i][j] = a_
        C_[i][j] = c_
        B[i][j] = b
        A[i][j] = a
        C[i][j] = c
    # defuzz = (1*A_ + 2*A + 3*B + 2*C + 1*C_) / (1+2+3+2+1)
    # print(defuzz)
# print(A)
# print(A_)
# print(B)
# print(C)
# print(C_)

L = len(datas) * k1
A = A.reshape((L,3))
A_ = A_.reshape((L,3))
B = B.reshape((L,3))
C = C.reshape((L,3))
C_ = C_.reshape((L,3))

# check constraint
# print(np.where(B<A))
# print(np.where(C<B))
# print(np.where(A_>A))
# print(np.where(C>C_))

# input image
input_data = readImage('images/8C0190 (1).jpg')
N = input_data.shape[0]

x = np.repeat(input_data, L, axis= 0)
a_matrix = np.vstack([A] * N)
b_matrix = np.vstack([B] * N)
c_matrix = np.vstack([C] * N)
a1_matrix = np.vstack([A_] * N)
c1_matrix = np.vstack([C_] * N)
# print(x.shape, a_matrix.shape)

# cal u, n, e
def triangular(x, a, a1, b, c, c1):
    u = np.zeros(x.shape)
    n = np.zeros(x.shape)
    e = np.zeros(x.shape)

    u[x <= a] = 0
    u[x >= c] = 0
    first_half = np.logical_and(a < x, x <= b)
    second_half = np.logical_and(b < x, x < c)
    u[first_half] = (x[first_half]-a[first_half]) / (b[first_half]-a[first_half])
    u[second_half] = (c[second_half]-x[second_half]) / (c[second_half]-b[second_half])

    f = np.zeros(x.shape)
    f[x <= a1] = 1
    f[x >= c1] = 1
    first_half = np.logical_and(a1 < x, x <= b)
    second_half = np.logical_and(b < x, x < c1)
    f[first_half] = (b[first_half]-x[first_half]) / (b[first_half]-a1[first_half])
    f[second_half] = (x[second_half]-b[second_half]) / (c1[second_half]-b[second_half])


    n = (1 - (1 - (u + f)) ** alpha) ** (1 / alpha) - u
    e = f - n

    return u, n, e

u, n, e = triangular(x, a_matrix, a1_matrix, b_matrix, c_matrix, c1_matrix)

# print(np.amax(u))
# print(np.amax(n))
# print(np.amax(e))

member_vals = u * (2 - e)
member_vals[member_vals > 1] = 1 

# ouput image
def max_rules(ms_matrix):
    ms_matrix = ms_matrix.reshape(N, L, 3)

    # mean of 3 rgb
    result = np.mean(ms_matrix, axis= 2)

    # max of rules
    result = np.amax(result, axis= 1)

    # show result grayscale image
    image = data2ImageGS(result)
    image.save('max_result.jpg')
    image.show()

def mean_rules(ms_matrix):
    ms_matrix = ms_matrix.reshape(N, L, 3)

    # mean of 3 rgb
    result = np.mean(ms_matrix, axis= 2)

    # max of rules
    result = np.mean(result, axis= 1)

    # show result grayscale image
    image = data2ImageGS(result)
    image.save('mean_result.jpg')
    image.show()

max_rules(member_vals)
mean_rules(member_vals)
