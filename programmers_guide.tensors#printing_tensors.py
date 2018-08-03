'''
(tensorflow) rm@ubuntu:~$ python
Python 3.6.5 |Anaconda, Inc.| (default, Apr 29 2018, 16:14:56) 
[GCC 7.2.0] on linux
Type "help", "copyright", "credits" or "license" for more information.

>>> #From https://www.tensorflow.org/programmers_guide/tensors#printing_tensors
... import tensorflow as tf

>>> p = tf.placeholder(tf.float32)
>>> t = p + 1.0
... print (t)  # This will print the symbolic tensor when the graph is being built.
Tensor("add:0", dtype=float32)

>>> tf.Print(t, [t])  # This does nothing
<tf.Tensor 'Print:0' shape=<unknown> dtype=float32>

>>> t = tf.Print(t, [t])  # Here we are using the value returned by tf.Print
>>> result = t + 1  # Now when result is evaluated the value of `t` will be printed.
>>> sess = tf.Session()
>>> with sess.as_default():
...     result.eval()
... 
Traceback (most recent call last):
...
...
  Caused by op 'Placeholder', defined at:
...
...
InvalidArgumentError (see above for traceback): You must feed a value for placeholder 
    tensor 'Placeholder' with dtype float
	 [[Node: Placeholder = Placeholder[dtype=DT_FLOAT, shape=<unknown>, _device="/job:localhost/replica:0/task:0/device:CPU:0"]()]]

>>> with sess.as_default():
...     result.eval(feed_dict={p:2.0})
... 
[3]
4.0
>>> 
>>> b = tf.Print(p, [t])  # Here we are using the value returned by tf.Print
>>> result = b + 1  # Now when result is evaluated the value of `t` will be printed.
>>> with sess.as_default():
...     result.eval(feed_dict={p:2.0})
... 
[3]
[3]
3.0
>>> 
'''
################################
#RUN THIS ON THE TERMINAL

import tensorflow as tf

p = tf.placeholder(tf.float32)
t = p + 1.0
#print (t)  # This will print the symbolic tensor when the graph is being built.
#Tensor("add:0", dtype=float32)
#Works on terminal not in SPYDER

#tf.Print(t, [t])  # This does nothing
#<tf.Tensor 'Print:0' shape=<unknown> dtype=float32>
#Works on terminal not in SPYDER

t = tf.Print(t, [t], message="\nprint_tensor_t: ")  # Here we are using the value returned by tf.Print
result = t + 1  # Now when result is evaluated the value of `t` will be printed.

sess = tf.Session()

with sess.as_default():
    print("result.eval(), p=1: ", result.eval(feed_dict={p:1.0}))
    print("sess.run of result = t+1, p=2: ", sess.run(result, feed_dict={p:2.0}))

#[3]
#4.0

b = tf.Print(p, [t], message="print_tensor_p: ")  # Here we are using the value returned by tf.Print
result = b + 1  # Now when result is evaluated the value of `t` will be printed.
with sess.as_default():
    print("result.eval(), p=3: ", result.eval(feed_dict={p:3.0}))
    print("sess.run of result = b+1, p=4: ", sess.run(result, feed_dict={p:4.0}))

#[3]
#[3]
#3.0

sess.close()

print("\n\tDONE: ", __file__)

