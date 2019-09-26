import numpy as np
import PIL
from PIL import Image
from os import listdir

size = (0,0)
def readImage(image_name):
    image = Image.open(image_name)
    size = (int(image.size[0]/5), int(image.size[1]/5))
    # print(size)
    resized_image = image.resize((size))
    resized_image.load()
    data = np.array(resized_image)
    data = image2vector(data)
    return data

# load all images in direc
def loadImageFromFile(directory):
    loaded_images = list()
    image_names = list()
    for filename in sorted(listdir(directory)):
        if filename == '.DS_Store':
            continue
        img_data = readImage(directory + '/'+ filename)
        # store loaded image
        loaded_images.append(img_data)
        image_names.append(filename)
        # print('> loaded %s %s' % (filename, img_data.shape))
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

def data2Image(data):
    image_data = data.reshape(data.shape[0], data.shape[1], 3)
    image = Image.fromarray(image_data.astype(np.uint8))
    return image

# data = readImage('test.jpg')
# data = data.reshape((384,512,3))
# image = data2Image(data)
# image.save('abc.jpg')