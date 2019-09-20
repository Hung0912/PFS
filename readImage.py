import numpy as np
import PIL
from PIL import Image
from os import listdir

size = 256, 256

def readImage(image_name):
    image = Image.open(image_name)
    resized_image = image.resize((size))
    resized_image.load()
    data = np.asarray(resized_image)
    data = image2vector(data)
    return data

# load all images in direc
def loadImageFromFile(directory):
    loaded_images = list()
    image_names = list()
    for filename in listdir(directory):
        if filename == '.DS_Store':
            continue
        img_data = readImage(directory + '/'+ filename)
        # store loaded image
        loaded_images.append(img_data)
        image_names.append(filename)
        print('> loaded %s %s' % (filename, img_data.shape))
    return loaded_images, image_names

def image2vector(image):
    """
    Argument:
    image -- a numpy array of shape (length, height, depth)
    
    Returns:
    v -- a vector of shape (length*height, depth)
    """
    v = image.reshape( image.shape[0] * image.shape[1], image.shape[2])
    return v

def vector2Image(data):
    image_data = data.reshape(256, 256, 3)
    image = Image.fromarray(image_data.astype(np.uint8))
    return image