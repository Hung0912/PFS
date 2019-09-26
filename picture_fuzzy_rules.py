import numpy as np
import random
import operator
import math
from picture_fuzzy_clustering import calculateClusterCenter

# box : (x,y), w, h
X_ = X[x:, y:]
#trich du lieu khu vuc co benh : data X_ va membership tuong ung matrix_
#tinh lai tam V cua tung khu vuc benh:

V_ = calculateClusterCenter(matrix_, X_)

#Ap dung luat mo retangle cho tung cum trong tung kv benh
u = matrix_[:,:,0]
n = matrix_[:,:,1]
e = matrix_[:,:,2]
u_ = (u + n) / (1 + e)

a1 = np.min(X_[:,0])
c1 = np.max(X_[:,0])
b = V_[:,0]
z = np.where( X_[:,0] <= b)
a = np.sum(np.reshape(u_, z.shape()) * z) / np.sum(np.reshape(u_, z.shape()))
y = np.where( X_[:,0] >= b)
c = np.sum(np.reshape(u_, y.shape()) * y) / np.sum(np.reshape(u_, y.shape()))


