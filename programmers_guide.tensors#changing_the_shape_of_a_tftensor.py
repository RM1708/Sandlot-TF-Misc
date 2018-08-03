'''
(tensorflow) rm@ubuntu:~$ python
Python 3.6.5 |Anaconda, Inc.| (default, Apr 29 2018, 16:14:56) 
[GCC 7.2.0] on linux
Type "help", "copyright", "credits" or "license" for more information.
>>> # From https://www.tensorflow.org/programmers_guide/tensors#changing_the_shape_of_a_tftensor
... import tensorflow as tf
>>> rank_three_tensor = tf.ones([3, 4, 5])
>>> matrix = tf.reshape(rank_three_tensor, [6, 10])  # Reshape existing content into
>>>                                                  # a 6x10 matrix
... matrixB = tf.reshape(matrix, [3, -1])  #  Reshape existing content into a 3x20
>>>                                        # matrix. -1 tells reshape to calculate
...                                        # the size of this dimension.
... matrixAlt = tf.reshape(matrixB, [4, 3, -1])  # Reshape existing content into a
>>>                                              #4x3x5 tensor
... sess = tf.Session()
>>> print(sess.run(rank_three_tensor))
[[[1. 1. 1. 1. 1.]
  [1. 1. 1. 1. 1.]
  [1. 1. 1. 1. 1.]
  [1. 1. 1. 1. 1.]]

 [[1. 1. 1. 1. 1.]
  [1. 1. 1. 1. 1.]
  [1. 1. 1. 1. 1.]
  [1. 1. 1. 1. 1.]]

 [[1. 1. 1. 1. 1.]
  [1. 1. 1. 1. 1.]
  [1. 1. 1. 1. 1.]
  [1. 1. 1. 1. 1.]]]
>>> print(sess.run(matrix))
[[1. 1. 1. 1. 1. 1. 1. 1. 1. 1.]
 [1. 1. 1. 1. 1. 1. 1. 1. 1. 1.]
 [1. 1. 1. 1. 1. 1. 1. 1. 1. 1.]
 [1. 1. 1. 1. 1. 1. 1. 1. 1. 1.]
 [1. 1. 1. 1. 1. 1. 1. 1. 1. 1.]
 [1. 1. 1. 1. 1. 1. 1. 1. 1. 1.]]
>>> print(sess.run(matrixB))
[[1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1.]
 [1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1.]
 [1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1.]]
>>> print(sess.run(matrixAlt))
[[[1. 1. 1. 1. 1.]
  [1. 1. 1. 1. 1.]
  [1. 1. 1. 1. 1.]]

 [[1. 1. 1. 1. 1.]
  [1. 1. 1. 1. 1.]
  [1. 1. 1. 1. 1.]]

 [[1. 1. 1. 1. 1.]
  [1. 1. 1. 1. 1.]
  [1. 1. 1. 1. 1.]]

 [[1. 1. 1. 1. 1.]
  [1. 1. 1. 1. 1.]
  [1. 1. 1. 1. 1.]]]

>>> print(sess.run(tf.reshape(matrixAlt, [13, 2, -1]))) #ERROR
Traceback (most recent call last):
...
...
ValueError: Dimension size must be evenly divisible by 26 but is 60 for 'Reshape_3' (op: 'Reshape') with input shapes: [4,3,5], [3] and with input tensors computed as partial shapes: input[1] = [13,2,?].
>>> 
(tensorflow) rm@ubuntu:~$ 

'''
import tensorflow as tf
rank_three_tensor = tf.ones([3, 4, 5])
matrix = tf.reshape(rank_three_tensor, [6, 10])  # Reshape existing content into
                                                 # a 6x10 matrix
matrixB = tf.reshape(matrix, [3, -1])  #  Reshape existing content into a 3x20
                                       # matrix. -1 tells reshape to calculate
                                       # the size of this dimension.
matrixAlt = tf.reshape(matrixB, [4, 3, -1])  # Reshape existing content into a
                                             #4x3x5 tensor
sess = tf.Session()
print("\nrank_three_tensor shape[3, 4, 5]:\n", sess.run(rank_three_tensor))
#[[[1. 1. 1. 1. 1.]
#  [1. 1. 1. 1. 1.]
#  [1. 1. 1. 1. 1.]
#  [1. 1. 1. 1. 1.]]
#
# [[1. 1. 1. 1. 1.]
#  [1. 1. 1. 1. 1.]
#  [1. 1. 1. 1. 1.]
#  [1. 1. 1. 1. 1.]]
#
# [[1. 1. 1. 1. 1.]
#  [1. 1. 1. 1. 1.]
#  [1. 1. 1. 1. 1.]
#  [1. 1. 1. 1. 1.]]]
print("\nmatrix shape[6, 10]:\n", sess.run(matrix))
#[[1. 1. 1. 1. 1. 1. 1. 1. 1. 1.]
# [1. 1. 1. 1. 1. 1. 1. 1. 1. 1.]
# [1. 1. 1. 1. 1. 1. 1. 1. 1. 1.]
# [1. 1. 1. 1. 1. 1. 1. 1. 1. 1.]
# [1. 1. 1. 1. 1. 1. 1. 1. 1. 1.]
# [1. 1. 1. 1. 1. 1. 1. 1. 1. 1.]]
print("\nmatrixB shape[3, -1]:\n", sess.run(matrixB))
#[[1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1.]
# [1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1.]
# [1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1.]]
print("\nmatrixAlt shape[4, 3, -1]: \n", sess.run(matrixAlt))
#[[[1. 1. 1. 1. 1.]
#  [1. 1. 1. 1. 1.]
#  [1. 1. 1. 1. 1.]]
#
# [[1. 1. 1. 1. 1.]
#  [1. 1. 1. 1. 1.]
#  [1. 1. 1. 1. 1.]]
#
# [[1. 1. 1. 1. 1.]
#  [1. 1. 1. 1. 1.]
#  [1. 1. 1. 1. 1.]]
#
# [[1. 1. 1. 1. 1.]
#  [1. 1. 1. 1. 1.]
#  [1. 1. 1. 1. 1.]]]

sess.close()
print("\n\tDONE: ", __file__)