'''
From 
https://stackoverflow.com/questions/47474869/getting-a-list-of-all-known-classes-of-vgg-16-in-keras
'''
from keras.applications.vgg16 import VGG16
from keras.preprocessing.image import load_img
from keras.preprocessing.image import img_to_array
from keras.applications.vgg16 import preprocess_input
from keras.applications.vgg16 import decode_predictions
print("Getting VGG16 pre-trained model ...")
model = VGG16()

print(model.summary())

print("Reading the image file ...")
image = load_img('/home/rm/tmp/Images/cat_hiding_face.jpeg', \
                 target_size=(224, 224))

print("Pre-processing the image data ...")
image = img_to_array(image)  #output Numpy-array

image = image.reshape((1, image.shape[0], image.shape[1], image.shape[2]))

image = preprocess_input(image)

print("Computing the class-ids ...")
class_id = model.predict(image)

print("Getting the Top 10 probable classes ...")
label = decode_predictions(class_id, top=10)[0]

[print("Class Name: {}; Probability: {}".format(label[i][1], \
           (int(label[i][2] * 1.0E04)/1.0E04))) for i in range(10)]
