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
    for filename in listdir(directory):
        if filename == '.DS_Store':
            continue
        resized_image = resizeImage(directory + '/'+ filename)
        # load image
        img_data = np.asarray(resized_image)
        # store loaded image
        loaded_images.append(img_data)
        # print('> loaded %s %s' % (filename, img_data.shape))
    return loaded_images

def readPixel(image):
    # matrix = [[0 for x in range(image.shape[0])] for y in range(image.shape[1])] 
    # matrix = np.array([[]])
    # print(matrix.size)
    for x in range(image.shape[0]):
        for y in range(image.shape[1]):
            print(image[x,y])

    # print(matrix)

# if __name__ == "__main__":
#     # resizeImage('images/8A0000.jpg')
#     loadImageFromFile('images')
