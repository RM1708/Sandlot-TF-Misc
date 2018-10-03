
# coding: utf-8

# # Intro
# (Downloaded from https://www.kaggle.com/kmader/transfer-learning-with-inceptionv3/code)
# 
# The notebook uses pretrained models of InceptionV3 and others (possibly) to try and predict the manufacturer of each camera, not sure why this is a good idea, but it's worth an experiment

# Copy weights into directories keras links



#get_ipython().system('mkdir ~/.keras')
#get_ipython().system('mkdir ~/.keras/models')
# not enough space for both
#!cp ../input/keras-pretrained-models/* ~/.keras/models/ 
#!cp ../input/vgg19/* ~/.keras/models
#get_ipython().system('cp ../input/keras-pretrained-models/*notop* ~/.keras/models/')
#get_ipython().system('cp ../input/keras-pretrained-models/imagenet_class_index.json ~/.keras/models/')
#get_ipython().system('cp ../input/keras-pretrained-models/resnet50* ~/.keras/models/')





import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import os
#from PIL import Image
#from skimage.transform import resize
from random import shuffle

from datetime import datetime

print("\n\tStart Time: ", datetime.time(datetime.now()))

print("Forming List of paths to the directories under ../input_data ...")
PROJ_DIR = "/home/rm/kaggle/IEEE-Camera-Contest/Inception_kmader/"
list_paths = []
for subdir, dirs, files in \
        os.walk(PROJ_DIR + "../input_data//"):
    for file in files:
        #print os.path.join(subdir, file)
        filepath = subdir + os.sep + file
        list_paths.append(filepath)




print("Making separate lists for train and test folders ...")
list_train = [filepath for filepath in list_paths if "train/" in filepath]
shuffle(list_train)
list_test = [filepath for filepath in list_paths if "test/" in filepath]

list_train = list_train[:1000] # ??
list_test = list_test[:100] # ??
index = [os.path.basename(filepath) for filepath in list_test]




#Each folder in train/ is the name of a class
list_classes = []
list_classes = list(set([os.path.dirname(filepath).split(os.sep)[-1] \
                         for filepath in list_paths if "train/" in filepath]))





#list_classes = ['Sony-NEX-7',
assert(set(list_classes) == set(['Sony-NEX-7',
                     'Motorola-X',
                     'HTC-1-M7',
                     'Samsung-Galaxy-Note3',
                     'Motorola-Droid-Maxx',
                     'iPhone-4s',
                     'iPhone-6',
                     'LG-Nexus-5x',
                     'Samsung-Galaxy-S4',
                     'Motorola-Nexus-6']))



ROWS=139
COLS=139
from keras.preprocessing.image import ImageDataGenerator
from keras.applications.inception_v3 import preprocess_input
train_idg = ImageDataGenerator(vertical_flip=True,
                               horizontal_flip=True,
                               height_shift_range=0.1,
                               width_shift_range=0.1,
                               preprocessing_function=preprocess_input)
train_gen = train_idg.flow_from_directory(
                                PROJ_DIR + \
                                "../input_data//train/",
                                target_size=(ROWS, COLS),
                                batch_size = 16)





from keras.models import Sequential
#from keras.models import Model
from keras.callbacks import ModelCheckpoint, \
                            EarlyStopping 
#                            TensorBoard, \
#                            LearningRateScheduler, \
#                            ReduceLROnPlateau, \
                            
from keras import optimizers \
#                    , losses, \
#                    activations, \
#                    models
from keras.layers import \
                            Dense, \
                            Dropout, \
                            GlobalAveragePooling2D
#                            Concatenate
#                            Convolution2D, \
#                            Input, \
#                            Flatten, \
#                            MaxPooling2D, \
#                            BatchNormalization, \
from keras import applications
input_shape = (ROWS, COLS, 3)
nclass = len(train_gen.class_indices)

print("Constructing the pre-trained part ...")
#https://keras.io/applications/#inceptionv3
#Weights have been trained on imagenet.
#
base_model = applications.InceptionV3(weights='imagenet', 
                                include_top=False, 
                                input_shape=(ROWS, COLS,3))
base_model.trainable = False

print("Adding the personalized layers ...")
add_model = Sequential()
add_model.add(base_model)
add_model.add(GlobalAveragePooling2D())
add_model.add(Dropout(0.5))
add_model.add(Dense(nclass, 
                    activation='softmax'))

model = add_model
model.compile(loss='categorical_crossentropy', 
              optimizer=optimizers.SGD(lr=1e-4, 
                                       momentum=0.9),
              metrics=['accuracy'])
print("Summarizing ...")
model.summary()





file_path="weights.best.hdf5"

checkpoint = ModelCheckpoint(file_path, monitor='acc', \
                             verbose=1, save_best_only=True, mode='max')

early = EarlyStopping(monitor="acc", mode="max", patience=15)

callbacks_list = [checkpoint, early] #early

print("Training the model ...")
history = model.fit_generator(train_gen, 
                              epochs=2, 
                              shuffle=True, 
                              verbose=True,
                              callbacks=callbacks_list)



print("Done Training.")

model.load_weights(file_path)





test_idg = ImageDataGenerator(preprocessing_function=preprocess_input)
test_gen = test_idg.flow_from_directory(
    PROJ_DIR + '../input_data/',
    target_size=(ROWS, COLS),
    batch_size = 16,
    shuffle = False,
    class_mode='binary',
    classes = ['test']
)
len(test_gen.filenames)




print("Testing ...")
predicts = model.predict_generator(test_gen, verbose = True, workers = 2)





predicts = np.argmax(predicts, 
                     axis=1)
label_index = {v: k for k,v in train_gen.class_indices.items()}
predicts = [label_index[p] for p in predicts]

df = pd.DataFrame(columns=['fname', 'camera'])
df['fname'] = [os.path.basename(x) for x in test_gen.filenames]
df['camera'] = predicts
df.to_csv("sub1.csv", index=False)

print("\n\tEnd Time: ", datetime.time(datetime.now()))

print("\n\tDONE: ", __file__)
