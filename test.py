import sys
sys.path.append('/usr/local/lib/python3.7/site-packages')
import csv
import numpy as np
n = 512*384
k = 8
matrix = np.zeros((n*k,3))
with open('results/membership_matrixs/8C0189 (3).csv') as f:
    readCSV = csv.reader(f, delimiter = ',')
    matrix = np.array(list(readCSV)).astype("float")
    
print(matrix.reshape((384,512,k,3)).shape)