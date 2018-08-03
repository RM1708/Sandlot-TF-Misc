###########################################
#From https://www.tensorflow.org/api_docs/python/tf/shape

import tensorflow as tf

t = tf.constant([[[1, 1, 1], [2, 2, 2]], [[3, 3, 3], [4, 4, 4]]])

shape = tf.shape(t)  # [2, 2, 3]
size = tf.size(t)  # 12

sess = tf.InteractiveSession()
print("Tensor t: \n", sess.run(t))
print("\nShape of t: ", sess.run(shape))
print("Size of t: ",sess.run(size))
sess.close()
