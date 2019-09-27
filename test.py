import sys
sys.path.append('/usr/local/lib/python3.7/site-packages')
import csv
import numpy as np
import skfuzzy as fuzz


x = np.arange(30)
x = x.reshape((5,3,2))

# print(np.where(x>=5, x, np.zeros((5,3,2))))
# print(np.where(x<=12, x, np.zeros((5,3,2))))
print(x>=5)
print(x<=12)