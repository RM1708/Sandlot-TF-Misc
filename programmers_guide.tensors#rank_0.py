'''
(tensorflow) rm@ubuntu:~$ python
Python 3.6.5 |Anaconda, Inc.| (default, Apr 29 2018, 16:14:56) 
[GCC 7.2.0] on linux
Type "help", "copyright", "credits" or "license" for more information.

# From: https://www.tensorflow.org/programmers_guide/tensors#rank_0

>>> import tensorflow as tf
>>> mammal = tf.Variable("Elephant", tf.string)
>>> ignition = tf.Variable(451, tf.int16)
>>> floating = tf.Variable(3.14159265359, tf.float64)
>>> its_complicated = tf.Variable(12.3 - 4.85j, tf.complex64)
>>> 
>>> sess = tf.Session()
>>> print(sess.run(mammal))
Traceback (most recent call last):
...
  raise type(e)(node_def, op, message)
tensorflow.python.framework.errors_impl.FailedPreconditionError: Attempting to 
    use uninitialized value Variable
	 [[Node: _retval_Variable_0_0 = _Retval[T=DT_STRING, index=0, _device="/job:localhost/replica:0/task:0/device:CPU:0"](Variable)]]

>>> sess.close()
>>> init = tf.global_variables_initializer()
>>> sess = tf.Session()
>>> sess.run(init)
>>> print(sess.run(mammal))
b'Elephant'

...

>>> print((sess.run(mammal)).decode("utf-8"))
Elephant

>>> print(sess.run(ignition))
451

>>> print(sess.run(floating))
3.1415927

>>> print(sess.run(its_complicated))
(12.3-4.85j)
>>> 
'''

import tensorflow as tf
mammal = tf.Variable("Elephant", tf.string)
ignition = tf.Variable(451, tf.int16)
floating = tf.Variable(3.14159265359, tf.float64)
its_complicated = tf.Variable(12.3 - 4.85j, tf.complex64)

sess = tf.Session()
init = tf.global_variables_initializer()
sess = tf.Session()
sess.run(init)
print(sess.run(mammal))
#b'Elephant'
print((sess.run(mammal)).decode("utf-8"))
#Elephant

print(sess.run(ignition))
#451

print(sess.run(floating))
#3.1415927

print(sess.run(its_complicated))
#(12.3-4.85j)

print("\n\tDONE: ", __file__)