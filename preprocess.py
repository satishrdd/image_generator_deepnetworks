import numpy as np
import pickle
from PIL import Image
from tensorflow.examples.tutorials.mnist import input_data
import os

def resize_images(image_arrays, size=[32, 32]):
    # convert float type to integer 
    image_arrays = (image_arrays * 255).astype('uint8')
    
    resized_image_arrays = np.zeros([image_arrays.shape[0]]+size)
    for i, image_array in enumerate(image_arrays):
        image = Image.fromarray(image_array)
        resized_image = image.resize(size=size, resample=Image.LANCZOS)
        
        resized_image_arrays[i] = np.asarray(resized_image)
    
    return np.expand_dims(resized_image_arrays, 3)  




mnist = input_data.read_data_sets("mnist/")


#get the training and testing samples

train = {'X': resize_images(mnist.train.images.reshape(-1, 28, 28)),'y': mnist.train.labels}

test = {'X': resize_images(mnist.test.images.reshape(-1, 28, 28)),'y': mnist.test.labels}

#remove saved file if already exists

try:
	os.remove('mnist/train.pkl')
	os.remove('mnist/test.pkl')
except OSError:
	print "Error in os.remove"
	exit(0)

#save into pickle
with open('mnist/train.pkl', 'wb') as f:
        pickle.dump(train, f, pickle.HIGHEST_PROTOCOL)
        print ('File Saved as: %s..' %path)
with open('mnist/test.pkl', 'wb') as f:
        pickle.dump(test, f, pickle.HIGHEST_PROTOCOL)
        print ('File Saved as: %s..' %path)   
