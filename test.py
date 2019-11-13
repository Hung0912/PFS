import sys
sys.path.append('/usr/local/lib/python3.7/site-packages')
import csv
import numpy as np

from PIL import Image

x = np.array([1,2,3,4,5,6,7,8,9,0,12,13])
y= np.reshape(x, (4,3))
print(y)

z = [1,2,3,2,3,2,3]
print(y[z])