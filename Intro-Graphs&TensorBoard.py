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

#for slicing error message
import sys, os

"""
3. # a rank 0 tensor; a scalar with shape [],
[1., 2., 3.] # a rank 1 tensor; a vector with shape [3]
[[1., 2., 3.], [4., 5., 6.]] # a rank 2 tensor; a matrix with shape [2, 3]
[[[1., 2., 3.]], [[7., 8., 9.]]] # a rank 3 tensor with shape [2, 1, 3]

"""
try:

    a = tf.constant(3.0, dtype=tf.float32)
    b = tf.constant(4.0) # also tf.float32 implicitly
    total = a + b
    
    print(a)
    print(b)
    print("a + b -> {}\n".format(total))
    """
    Expected Output
    
    Tensor("Const:0", shape=(), dtype=float32)
    Tensor("Const_1:0", shape=(), dtype=float32)
    Tensor("add:0", shape=(), dtype=float32)
    
    """
    
    writer = tf.summary.FileWriter('.')
    writer.add_graph(tf.get_default_graph())
    
    """
    This will produce an event file in the current directory with a name in the following format:
    
    events.out.tfevents.{timestamp}.{hostname}
    
    """
    
except(TypeError) as err:
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
