'''
(tensorflow) rm@ubuntu:~$ python
Python 3.6.5 |Anaconda, Inc.| (default, Apr 29 2018, 16:14:56) 
[GCC 7.2.0] on linux
Type "help", "copyright", "credits" or "license" for more information.
>>> #From https://www.tensorflow.org/programmers_guide/tensors#evaluating_tensors

>>> import tensorflow as tf
>>> constant = tf.constant([1, 2, 3])
>>> tensor = constant * constant
>>> sess = tf.Session()

>>> print(tensor.eval)
<bound method Tensor.eval of <tf.Tensor 'mul:0' shape=(3,) dtype=int32>>

>>> print(tensor.eval())
Traceback (most recent call last):
...
...
ValueError: Cannot evaluate tensor using `eval()`: No default session is registered. Use `with sess.as_default()` or pass an explicit session to `eval(session=sess)`

>>> print(sess.run(tensor.eval()))
Traceback (most recent call last):
...
...
ValueError: Cannot evaluate tensor using `eval()`: No default session is registered. Use `with sess.as_default()` or pass an explicit session to `eval(session=sess)`

>>> print(sess.run(tensor))
[1 4 9]

>>> 
>>> sess = tf.Session()
>>> with sess.as_default():
... print(tensor.eval())
  File "<stdin>", line 2
    print(tensor.eval())
        ^
IndentationError: expected an indented block
>>> with sess.as_default():
...     with sess.as_default():
...             with sess.as_default():
...                     print(tensor.eval())
...             print(tensor.eval())
...     print(tensor.eval())
... 
[1 4 9]
[1 4 9]
[1 4 9]
>>> 
>>> with sess.as_default():
...     p = tf.placeholder(tf.float32)
... t = p + 1.0
  File "<stdin>", line 3
    t = p + 1.0
    ^
SyntaxError: invalid syntax

>>> p = tf.placeholder(tf.float32)
>>> t = p + 1.0
>>> with sess.as_default():
...     t.eval()  # This will fail, since the placeholder did not get a value.
...     t.eval(feed_dict={p:2.0})  # This will succeed because we're feeding a value
...                            # to the placeholder.
... 
Traceback (most recent call last):
...
...
Caused by op 'Placeholder', defined at:
...
...
InvalidArgumentError (see above for traceback): You must feed a value for placeholder tensor 'Placeholder' with dtype float
	 [[Node: Placeholder = Placeholder[dtype=DT_FLOAT, shape=<unknown>, _device="/job:localhost/replica:0/task:0/device:CPU:0"]()]]


>>> with sess.as_default():
...     t.eval(feed_dict={p:2.0})  # This will succeed because we're feeding a value
...                            # to the placeholder.
... 
3.0
>>> 
'''
import tensorflow as tf

constant = tf.constant([1, 2, 3])
const_sqrd = constant * constant


sess = tf.Session()

with sess.as_default():
    print("const_sqrd.eval(): ", const_sqrd.eval())
with sess.as_default():
    with sess.as_default():
        with sess.as_default():
            print("\t\tconst_sqrd.eval(): ", const_sqrd.eval())
        print("\tconst_sqrd.eval(): ", const_sqrd.eval())
    print("const_sqrd.eval(): ", const_sqrd.eval())
 
#[1 4 9]
#[1 4 9]
#[1 4 9]

p = tf.placeholder(tf.float32)
t = p + 1.0
with sess.as_default():
     print("\nt.eval(): ",t.eval(feed_dict={p:2.0}))  # This will succeed because we're feeding a value
                            # to the placeholder.
#3.0
                            
print("\n\tDONE: ", __file__)