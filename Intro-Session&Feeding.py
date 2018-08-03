# -*- coding: utf-8 -*-
"""
https://www.tensorflow.org/programmers_guide/low_level_intro

This was run in Spyder IDE
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import tensorflow as tf

#for slicing error message
import sys, os

a = tf.constant(3.0, dtype=tf.float32)
b = tf.constant(4.0) # also tf.float32 implicitly
total = a + b
try:

    sess = tf.Session()
    print("sess.run(total): {}\n".format(sess.run(total)))
    
    #You can pass multiple tensors to tf.Session.run. 
    #The run method transparently handles any combination 
    #of tuples or dictionaries, as in the following example:
    
    print(sess.run({'ab':(a, b), 'total':total}), "\n")
    
    #which returns the results in a structure of the same layout:
    #{'total': 7.0, 'ab': (3.0, 4.0)}
    
    #During a call to tf.Session.run any tf.Tensor only has a single value. 
    #For example, the following code calls tf.random_uniform to 
    #produce a tf.Tensor that generates a random 3-element vector 
    #(with values in [0,1)):
    
    #The result shows a different random value on each call to run, 
    #but a consistent value during a single run 
    #(out1 and out2 receive the same random input):
    
    vec = tf.random_uniform(shape=(3,))
    out1 = vec + 1
    out2 = vec + 2
    result = sess.run(vec)
    print("result: {}".format(result))
    print("%7.4f, %7.4f, %7.4f"% (result[0], result[1], result[2]))

    print(sess.run(vec))
    out1_, out2_ = sess.run((out1, out2))
    print("\nout1: {} \nout2: {}".format(out1_, out2_))
    
    x = tf.placeholder(tf.float32)
    y = tf.placeholder(tf.float32)
    
    z = x + y

    print("\n")
    print(sess.run(z, feed_dict={x: 3, y: 4.5}))
    print(sess.run(z, feed_dict={x: [1, 3], y: [2, 4]}))
    print(sess.run(z, feed_dict={x: [[1, 3], [5, 6]], \
                                 y: [[7, 8], [9, 10]]}))

except(TypeError, ValueError) as err:
    print("\nERROR ERROR ERROR:")
    print("------------------- ")
    print(err)
    """
    from:
    https://stackoverflow.com/questions/1278705/python-when-i-catch-an-exception-how-do-i-get-the-type-file-and-line-number
    """
    exc_type, exc_obj, exc_tb = sys.exc_info()
    fname = os.path.split(exc_tb.tb_frame.f_code.co_filename)[1]
    print("\n\tException Type: ",exc_type,"; \n\tIn File: ", \
          fname, "; ", "\n\tAt Line No: ",exc_tb.tb_lineno)  
    
    sess.close() 
else:
    sess.close()
