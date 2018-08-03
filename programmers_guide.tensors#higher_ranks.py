'''
(tensorflow) rm@ubuntu:~$ python
Python 3.6.5 |Anaconda, Inc.| (default, Apr 29 2018, 16:14:56) 
[GCC 7.2.0] on linux
Type "help", "copyright", "credits" or "license" for more information.
>>> #From: https://www.tensorflow.org/programmers_guide/tensors#higher_ranks
...
...

>>> import tensorflow as tf
>>> mymat = tf.Variable([[7],[11]], tf.int16)
>>> myxor = tf.Variable([[False, True],[True, False]], tf.bool)
>>> linear_squares = tf.Variable([[4], [9], [16], [25]], tf.int32)
>>> squarish_squares = tf.Variable([ [4, 9], [16, 25] ], tf.int32)
>>> rank_of_squares = tf.rank(squarish_squares)
>>> mymatC = tf.Variable([[7],[11]], tf.int32)

>>> sess = tf.Session()
>>> sess.run(tf.global_variables_initializer())

>>> print(sess.run(mymat))
[[ 7]
 [11]]
>>> print(sess.run(mymat)[0])
[7]
>>> print(sess.run(mymat)[0, 0])
7
>>> print(sess.run(mymat)[0][0])
7
>>> print(sess.run(myxor))
[[False  True]
 [ True False]]
>>> print(sess.run(myxor)[1])
[ True False]
>>> print(sess.run(myxor)[1,1])
False
>>> print(sess.run(linear_squares))
[[ 4]
 [ 9]
 [16]
 [25]]
>>> print(sess.run(linear_squares)[3, 0])
25
>>> print(sess.run(squarish_squares))
[[ 4  9]
 [16 25]]
 
>>> print(sess.run(squarish_squares).rank)
Traceback (most recent call last):
...
...
>>> import numpy as np
>>> print(sess.run(squarish_squares).shape)
(2, 2)
>>> print(sess.run(rank_of_squares))
2
>>> print(sess.run(mymatC))
[[ 7]
 [11]]
>>> sess.close()
>>> 

'''

import tensorflow as tf
mymat = tf.Variable([[7],[11]], tf.int16)
myxor = tf.Variable([[False, True],[True, False]], tf.bool)
linear_squares = tf.Variable([[4], [9], [16], [25]], tf.int32)
squarish_squares = tf.Variable([ [4, 9], [16, 25] ], tf.int32)
rank_of_squares = tf.rank(squarish_squares)
mymatC = tf.Variable([[7],[11]], tf.int32)

sess = tf.Session()
sess.run(tf.global_variables_initializer())

print(sess.run(mymat))
#[#[ 7]
 #[11]]
print(sess.run(mymat)[0])
#[7]
print(sess.run(mymat)[0, 0])
#7
print(sess.run(mymat)[0][0])
#7
print(sess.run(myxor))
#[#[False  True]
 #[ True False]]
print(sess.run(myxor)[1])
#[ True False]
print(sess.run(myxor)[1,1])
#False
print(sess.run(linear_squares))
#[#[ 4]
 #[ 9]
 #[16]
 #[25]]
print(sess.run(linear_squares)[3, 0])
#25
print(sess.run(squarish_squares))
#[#[ 4  9]
 #[16 25]]
 
print(sess.run(squarish_squares).shape)
#(2, 2)
print(sess.run(rank_of_squares))
#2
print(sess.run(mymatC))
#[#[ 7]
 #[11]]
 
sess.close()

print("\n\tDONE: ", __file__)