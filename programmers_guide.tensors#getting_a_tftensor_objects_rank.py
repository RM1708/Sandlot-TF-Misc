'''
(tensorflow) rm@ubuntu:~$ python 
Python 3.6.5 |Anaconda, Inc.| (default, Apr 29 2018, 16:14:56) 
[GCC 7.2.0] on linux
Type "help", "copyright", "credits" or "license" for more information.
>>> #From https://www.tensorflow.org/programmers_guide/tensors#getting_a_tftensor_objects_rank
... import tensorflow as tf
>>> my_image = tf.zeros([10, 299, 299, 3])  # batch x height x width x color
>>> sess = tf.Session()
>>> sess.close()
>>> r = tf.rank(my_image)
>>> # After the graph runs, r will hold the value 4.
... 
>>> sess = tf.Session()
>>> print(sess.run(r))
4
>>>
'''

import tensorflow as tf
my_image = tf.zeros([10, 299, 299, 3])  # batch x height x width x color
r = tf.rank(my_image)
# After the graph runs, r will hold the value 4.

sess = tf.Session()
print(sess.run(r))
#4
sess.close()

print("\n\tDONE: ", __file__)