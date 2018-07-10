#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jul  8 23:06:39 2018

@author: rm
"""

#for slicing error message
import sys, os

#import tensorflow
import tensorflow as tf
#open a variable scope named 'scope1'
try:
    with tf.variable_scope("scope1"):
        #declare a variable named variable1
        var1 = tf.get_variable("variable1",[1])
        #declare another variable with same name
        var2=tf.get_variable("variable1",[1])
except:
    print("EXCEPTION: ")    
    """
    from:
    https://stackoverflow.com/questions/1278705/python-when-i-catch-an-exception-how-do-i-get-the-type-file-and-line-number
    """
    exc_type, exc_obj, exc_tb = sys.exc_info()
    fname = os.path.split(exc_tb.tb_frame.f_code.co_filename)[1]
    print("Exception thrown. \n\tException Type: ", exc_type,\
          "; \n\tIn File: ", fname, "; ", \
          "\n\tAt Line No: ",exc_tb.tb_lineno)  
    #Repeated running will trigger exception at line 18, 
    #the first declaration
    tf.reset_default_graph()

