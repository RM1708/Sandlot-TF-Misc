#From https://stackoverflow.com/questions/42438170/tensorflow-using-a-session-graph-in-method. 
# See:  answered Feb 27 '17 at 13:43 by jabalazs

import tensorflow as tf
import numpy as np

class ImageAugmenter(object):
    def __init__(self, sess):
        self.sess = sess
        self.im_placeholder = tf.placeholder(tf.float32, shape=[1,784,3])

    def augment(self, image):
        #Define the augmentation operation graph node
        augment_op = tf.image.random_saturation(self.im_placeholder, 0.6, 0.8)
        return self.sess.run(augment_op, {self.im_placeholder: image})

class ListOfImages(object):
    def __init__(self, data_dir, sess):
        #NOTE: No "forward declaration" needed for load_data(). The interpreter "sees" the function
        # which is "declared" below.
        #
        # The image data is read and kept in the objects images attribute
        self.images = load_data(data_dir)
        
        #The ImageAugmenter object is created and is made available to the ListOfImages object in its augmenter attribute.
        # The augmenter 
        #       1. Is made aware of the session, by passing it in the constructor
        #       2. In its __init__ method, it creates a placeholder to hold the image data that it is expected to augment
        self.augmenter = ImageAugmenter(sess)

    def process_data(self):
        # This method applies the required augmentation (defined in the class ImageAugmenter).
        # It is applied to the data that was loaded in the ListOfImages object that called the method.
        # The data that was loaded is in the attribute images
        processed_images = []
        for im in self.images:
            processed_images.append(self.augmenter.augment(im))
        return processed_images

def load_data(data_dir):
    # True method would read images from disk
    # This is just a mockup
    images = []
    images.append(np.random.random([1,784,3]))
    images.append(np.random.random([1,784,3]))
    return images

if __name__ == "__main__":
    IMAGE_DATA_DIR = '/tmp/data/'
    sess = tf.Session()
    image_data = ListOfImages(IMAGE_DATA_DIR, sess)
    ListOfAugmentedImages = image_data.process_data()
    print(ListOfAugmentedImages)
    sess.close()


