import sys
sys.path.append('/usr/local/lib/python3.7/site-packages')
import cv2
import numpy as np
import csv
from readImage import *
from picture_fuzzy_clustering import n,k
size = (512,384)
n = size[0] * size[1]

def read_matrixCSV(image_name):
    matrix = np.zeros((n*k,3))
    with open('results/membership_matrixs/' + image_name[:-4] + '.csv') as f:
        readCSV = csv.reader(f, delimiter = ',')
        matrix = np.asarray(list(readCSV), dtype = float)
    
    return matrix.reshape((size[1],size[0],k,3))

def cropper():
    cropped_datas = list()
    cropped_matrixs = list()
    print('cropping ...')
    loaded_images, image_names = loadImageFromFile('images')
    with open('crop.csv') as f:
        readCSV = csv.reader(f, delimiter = ',')
        for row in readCSV:
            index = int(row[4])
            x = int(int(row[0]) / 5)
            y = int(int(row[1]) / 5)
            width = int(int(row[2])/5)
            height = int(int(row[3])/5)
            data = loaded_images[index]
            cropped_data = data.reshape(size[1],size[0],3)[y:y+height,x:x+width,:]
            cropped_datas.append(cropped_data)
            matrix = read_matrixCSV(image_names[index])
            cropped_matrix = matrix[y:y+height,x:x+width,:,:]
            print(cropped_matrix.shape)
            cropped_matrixs.append(cropped_matrix)
    print("crop done!")
    return cropped_datas, cropped_matrixs
    
# def save_cropper_toCSV(datas, matrixs):
#     with open('cropper/' + 'datas.csv', "w+") as f:
#         csv_write = csv.writer(f, delimiter = ',')
#         for i in range(len(datas)):
#             csv_write.writerows(datas[i])
    
#     with open('cropper/' + 'matrixs.csv', "w+") as f:
#         csv_write = csv.writer(f, delimiter = ',')
#         for i in range(len(matrixs)):
#             csv_write.writerows(matrixs[i])

# if __name__ == "__main__":
#     datas, matrixs = cropper()
#     save_cropper_toCSV(datas, matrixs)