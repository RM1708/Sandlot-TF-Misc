#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jul 23 21:58:05 2018

@author: rm
"""
import tensorflow as tf
import numpy as np

batch_size = 3
no_of_scanlines_per_image = 2
no_of_pixels_per_scanline = 4
all_hidden_states = [
                        [[1, 2, 3, 4],
                         [4, 5, 6, 7]],
                        
                        [[7, 8, 9, 10],
                         [10, 11, 12, 13]],
                        
                        [[13, 14, 15, 16],
                         [16, 17, 18, 17]]
                    ]
all_hidden_states = np.asarray(all_hidden_states)
########################################################################
#This block is to illustrate a conditional selection of input tensor
def true_fn():
    return (all_hidden_states)

def false_fn():
    z = tf.transpose(all_hidden_states,perm=[1,0,2])
    return (z)

cond_T_all_hidden_states = tf.cond((tf.convert_to_tensor(1 < 2)), \
                                 true_fn, \
                                 false_fn, \
                                 name="cond_T_all_hidden_states")
cond_F_all_hidden_states = tf.cond((tf.convert_to_tensor(1 < 0)), \
                                 true_fn, \
                                 false_fn, \
                                 name="cond_F_all_hidden_states")
#########################################################################
#
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())


#        This checks the True branch of the tf.cond() node.
#        It returns the tensor, all_hidden_states, 
#        if the condition holds True
    x = sess.run(cond_T_all_hidden_states)
    assert(batch_size == x.shape[0] and
           no_of_scanlines_per_image == x.shape[1] and
           no_of_pixels_per_scanline == x.shape[2])
    
#        This checks the False branch of the tf.cond() node.
#        It returns the tensor transpose [1,0,2] of, all_hidden_states, 
#        if the condition fails
    x = sess.run(cond_F_all_hidden_states)
    assert(no_of_scanlines_per_image == x.shape[0] and
           batch_size == x.shape[1] and
           no_of_pixels_per_scanline == x.shape[2])
    
    print("\n\tDONE: ", __file__)