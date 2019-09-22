import numpy as np
n = 256*256
k= 5
data = np.zeros((n,3))
cluster_centers = np.ones((n,1))
x = data / cluster_centers
print(x.shape)