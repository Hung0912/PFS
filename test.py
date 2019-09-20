import numpy as np
a = np.array([2,3,4,5])
result = np.where(a == np.amax(a))
print(result[0][0])