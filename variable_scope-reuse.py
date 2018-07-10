#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jul  8 22:50:04 2018

@author: rm
"""

#import tensorflow
import tensorflow as tf
#open a variable scope named 'scope1'
with tf.variable_scope("scope1"):
    #declare a variable named variable1
    var1 = tf.get_variable("variable1",[1])
    
    #set reuse flag to True
    tf.get_variable_scope().reuse_variables()
    #just an assertion!
    assert tf.get_variable_scope().reuse==True
    
    #declare another variable with same name
    var2=tf.get_variable("variable1",[1])

assert((var1==var2))

tf.reset_default_graph()
