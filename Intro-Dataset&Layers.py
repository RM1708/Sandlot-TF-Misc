# -*- coding: utf-8 -*-
"""
https://www.tensorflow.org/programmers_guide/low_level_intro

Run in Spyder IDE
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import tensorflow as tf

#Datasets
#https://www.tensorflow.org/api_docs/python/tf/data
my_data = [[0, 1,], [2, 3,], [4, 5,], [6, 7,],]

slices = tf.data.Dataset.from_tensor_slices(my_data)
next_item = slices.make_one_shot_iterator().get_next()

sess = tf.Session()

while True:
  try:
    print("Next Item: ", sess.run(next_item))
  except tf.errors.OutOfRangeError:
    print("\n\tError: Index Out Of Range")
    break
print("\n")
while True:
  try:
    print("Next Item: ", sess.run(next_item))
  except tf.errors.OutOfRangeError:
    print("\n\tError: Index Out Of Range")
    break
print("\n")

#If the Dataset depends on stateful operations you may need 
#to initialize the iterator before using it, 

r = tf.random_normal([10,3])
#print(r)
dataset = tf.data.Dataset.from_tensor_slices(r)
iterator = dataset.make_initializable_iterator()
next_row = iterator.get_next()

sess.run(iterator.initializer)
while True:
  try:
    print(sess.run(next_row))
  except tf.errors.OutOfRangeError:
    print("\n\tError: Index Out Of Range")
    break
print("\n")

sess.run(iterator.initializer)
while True:
  try:
    print(sess.run(next_row))
  except tf.errors.OutOfRangeError:
    print("\n\tError: Index Out Of Range")
    break
print("\n")

#For more details on Datasets and Iterators see: 
# https://www.tensorflow.org/programmers_guide/datasets
    
#Layers
#Layers package together both the variables and the operations 
#that act on them. For example a densely-connected layer
#(https://developers.google.com/machine-learning/glossary/#fully_connected_layer) 
#performs a weighted sum across all inputs for each output and 
#applies an optional activation function.
#(https://developers.google.com/machine-learning/glossary/#activation_function) 
#The connection weights and biases are managed by the layer object.

x = tf.placeholder(tf.float32, shape=[None, 3])
linear_model = tf.layers.Dense(units=5)
y = linear_model(x)

#Initializing Layers
init = tf.global_variables_initializer()
sess.run(init)
#note that this global_variables_initializer only initializes 
#variables that existed in the graph when the initializer was created.
#So the initializer should be one of the last things added during graph construction.

z = (sess.run(y, {x: [[1, 2, 3], [4, 5, 6]]}))
print("z: \n", z)
print("z.shape", z.shape)
print("\n")

y = tf.layers.dense(x, units=5)

init = tf.global_variables_initializer()
sess.run(init)

z = (sess.run(y, {x: [[1, 1, 1], [1, 1, 1]]}))
print("z: \n", z)
print("z.shape", z.shape)
print("\n")




