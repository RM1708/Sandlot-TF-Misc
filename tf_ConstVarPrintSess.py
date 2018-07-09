# coding: utf-8
"""
Run in Spyder IDE

"""

import tensorflow as tf
a = tf.placeholder("int32")
b = tf.placeholder("int32")
y = tf.multiply(a, b)

x = tf.Variable( "int32", name="var_x")

sess = tf.Session()

sess.run(tf.global_variables_initializer())

result = (sess.run(y, feed_dict={a: 2, b: 6}))
print(result)
x = sess.run(a, feed_dict={a: 3})
y = sess.run(b, feed_dict={a:3, b: 7})
print("a: ",x, "b: ", y)
sess.close()
#get_ipython().run_line_magic('save', 'tmp.py')
#get_ipython().run_line_magic('pwd', '')
