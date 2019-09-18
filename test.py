import numpy as np
import operator
x = 0
def distance(x, y):
    return np.linalg.norm(list(map(operator.sub, x, y)))

a = 9
print(np.square(a))
