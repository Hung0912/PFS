import sys
sys.path.append('/usr/local/lib/python3.7/site-packages')
import csv
import numpy as np

from PIL import Image

x = [1 , 0.5, 2, 0.4, 0.1, 1.5]
y = np.reshape(x, (2,3))
y[y >1] = 1
print(y)