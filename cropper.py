import sys
sys.path.append('/usr/local/lib/python3.7/site-packages')
import cv2
import numpy as np
import csv
from readImage import *
from picture_fuzzy_clustering import *


def read_matrixCSV(image_name):
    matrix = np.zeros((n*K,3))
    with open('results/membership_matrixs/' + image_name[:-4] + '.csv') as f:
        readCSV = csv.reader(f, delimiter = ',')
        matrix = np.asarray(list(readCSV), dtype = float)
    return matrix.reshape((size[1],size[0],K,3))

def cropper():
    cropped_datas = list()
    cropped_matrixs = list()
    print('cropping ...')
    loaded_images, image_names = loadImageFromFile('images')
    with open('crop.csv') as f:
        readCSV = csv.reader(f, delimiter = ',')
        i = 0
        for row in readCSV:
            index = int(row[4])
            x = int(int(row[0]) / 5)
            y = int(int(row[1]) / 5)
            width = int(int(row[2])/5)
            height = int(int(row[3])/5)
            data = loaded_images[index]
            tmp = np.zeros((384,512))
            tmp[y:y+height,x:x+width] = 1
            tmp = np.reshape(tmp,(384*512))
            # print(tmp.shape)
            # print(tmp)
            with open( 'cropper/tmps/tmp' + str(i) + '.csv', "w+") as f:
                csv_write = csv.writer(f, delimiter = ',')
                csv_write.writerow(tmp)

              
            cropped_data = data.reshape(size[1],size[0],3)[y:y+height,x:x+width,:]
            # print(cropped_data.shape)
            cropped_datas.append(cropped_data)
            
            matrix = read_matrixCSV(image_names[index])
            cropped_matrix = matrix[y:y+height,x:x+width,:,:]
            cropped_matrixs.append(cropped_matrix)
            # print(cropped_matrix.shape)
            save_cropper(cropped_data, cropped_matrix, i)
            i += 1
    print("crop done!")
    return cropped_datas, cropped_matrixs
    
def save_cropper(data, matrix, i):
    image = data2Image(data)
    image.save('cropper/datas/crop' + str(i) + '.jpg')
    matrix = np.reshape(matrix, (matrix.shape[0] * matrix.shape[1], matrix.shape[2], matrix.shape[3]))
    with open( 'cropper/matrixs/' + str(i) + '.csv', "w+") as f:
        csv_write = csv.writer(f, delimiter = ',')
        for j in range(matrix.shape[0]):
            csv_write.writerows(matrix[j])
    
if __name__ == "__main__":
    datas, matrixs = cropper()