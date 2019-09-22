import numpy as np

m = np.arange(8).reshape(5,3,2)
n = np.arange(8).reshape(5,3,2)

print(np.linalg.norm(m-n, axis=(1,2)))
