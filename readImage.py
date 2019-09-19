import PIL
from PIL import Image
import numpy as np
from matplotlib import image
from os import listdir

# Use Pillow
# print("Pillow version:", PIL.__version__)

size = 256, 256

def resizeImage(imageName):
    image = Image.open(imageName)
    # print('Format:', image.format, ', Mode:',
        #   image.mode, ', Size:', image.size)
    
    resized_image = image.resize((size))
    # data = np.asarray(resized_image)
    # image2 = Image.fromarray(data)
    # image2.show()
    return resized_image

# load all images in direc
def loadImageFromFile(directory):
    loaded_images = list()
    image_names = list()
    for filename in listdir(directory):
        if filename == '.DS_Store':
            continue
        resized_image = resizeImage(directory + '/'+ filename)
        # load image
        img_data = np.asarray(resized_image)
        # store loaded image
        loaded_images.append(img_data)
        image_names.append(filename)
        # print('> loaded %s %s' % (filename, img_data.shape))
    return loaded_images, image_names

# def to2dArray(image):
#     # matrix = [[0 for x in range(image.shape[0])] for y in range(image.shape[1])] 
#     # matrix = np.array([[]])
#     # print(matrix.size)
#     data = list()
#     for x in range(image.shape[0]):
#         for y in range(image.shape[1]):
#             data.append(image[x,y])
#     return data

def to2dArray(image):
    array = np.asarray(image)
    array = array.reshape(image.shape[0]*image.shape[1], image.shape[2])
    return array

def normalizeData(data):
    array = np.true_divide(data, 255)
    return array

def from2dArrayToImage(list_):
    list_ = np.array(list_).reshape(-1, 3)
    image_data = np.array(list_).reshape(256, 256, 3)
    image = Image.fromarray(image_data.astype(np.uint8))
    return image

# if __name__ == "__main__":
#     # resizeImage('images/8A0000.jpg')
#     loaded_image, image_names = loadImageFromFile('images')
    
#     data = to2dArray(loaded_image[0])
#     print(data[0])
#     data = normalizeData(data)
#     print(data[0])
#     # image = from2dArrayToImage(data)
#     # image.show()
