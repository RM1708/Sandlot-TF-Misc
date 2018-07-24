# -*- coding: utf-8 -*-
"""
Refactored from vanilla_rnn_with_tfboard.py.

Purpose
-------
Separate out the  bunch of asserts that were asserting mmy understanding
RM:
    1. Explanatory notes and comments.
    2. Purpose_indicator naming
"""
#from __future__ import print_function
import tensorflow as tf
import numpy as np

# Import MNIST data
from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("/home/rm/tmp/data/", one_hot=True)

# Define some parameters
no_of_elements_per_scanline = 28
no_of_scanlines_per_image = 28
no_of_classes = 10
batch_size = 128

NUM_OF_TRG_ITERS = 3000
CHECK_EVERY_ITER_NUM = 100

#no_of_hidden_layers is a misnomer. 
#The number gives the number of dimensions 
#(the dimensionality) of the state vector. 
#The state vector is taken as a row vector. The no of 
#components of this vector equals no_of_hidden_layers.
#More appropriate name is state_vector_dimensionality.
#
#The dimensionality is the number of ***independent*** features that the
#convolutional layer can detect

#<QUOTE>
#... in typical CNN models we stack convolutional layers hierarchically, 
#and feature map is simply a commonly used term referring to the output 
#of each such layer. Another way to view the output of these layers 
#is as processed images, the result of applying a filter and perhaps 
#some other operations. Here, ***this filter*** is parameterized by W, the 
#learned weights of our network representing the convolution filter.
#</QUOTE> (emphasis mine)
#
#From: "Learning TensorFlow: A Guide to Building Deep Learning Systems" 
#(Kindle Locations 1378-1382). 

#<QUOTE>
#This means that all the neurons in the first hidden layer will recognize 
#the same features, just placed differently in the input image. 
#For this reason, the map of connections from the input layer to the 
#hidden feature map ... . Obviously, we need to recognize an image of 
#more than a map of features, so *** a *** complete convolutional layer 
#is made from *** multiple *** feature maps.
#</QUOTE>
#
#From:G"etting Started with TensorFlow 
#(Kindle Locations 2065-2066). 

#no_of_hidden_layers = 128
state_vector_dimensionality = 128//2

# Where to save TensorBoard model summaries
LOG_DIR = "/home/rm/logs/RNN_with_summaries/"

# Create placeholders for inputs, labels
_input_images = tf.placeholder(tf.float32,
                         shape=[None, \
                                no_of_scanlines_per_image, \
                                no_of_elements_per_scanline],
                         name='inputs')
labels_true = tf.placeholder(tf.float32, shape=[None, \
                                      no_of_classes], \
                                name='labels')

########################################################################
# Weights and bias for input and hidden layer
#        Wx is the Weight matrix that post-multiplies the input vector.
#        Each input is a line of pixels. It is thus a row vector having
#        a dimensionality of no_of_elements_per_scanline.
#        Thus post multiplying it with Wx produces a row matrix with
#        no of elements equal to state_vector_dimensionality.
#        This resultant row matrix is the incremental change in the state 
#        vector state caused by the current input. It is to be added to
#        time-evolved state vector. That evolution is described next.
Wx = tf.Variable(tf.zeros([no_of_elements_per_scanline, \
                           state_vector_dimensionality]))

#        Wh is the weight matrix that post multiplies the state vector 
#        (a row vector). This results in the time evolution of the state vector
#        to the current instatnt of time.
#        As mentioned the state vector is a row vector having elements/components
#        whose number equals state_vector_dimensionality. So we need a 
#        transformation matrix that takes the input vector and combines its 
#        components in linearly independent(?) manner resulting in another
#        vector having the same number of elements/components. This transformation 
#        matrix, therefore, is a 2-D square matrix of 
#        shape [state_vector_dimensionality, state_vector_dimensionality]
Wh = tf.Variable(tf.zeros([state_vector_dimensionality, \
                           state_vector_dimensionality]))
        
b_rnn = tf.Variable(tf.zeros([state_vector_dimensionality]))
########################################################################
#NOTE: No scoping here
# Processing inputs to work with scan function
# Current input shape: (batch_size, no_of_scanlines_per_image, no_of_elements_per_scanline)
transposed_input = tf.transpose(_input_images, perm=[1, 0, 2])
# Current input shape now: (no_of_scanlines_per_image,batch_size, no_of_elements_per_scanline)
assert_transposed_0 = tf.Assert(tf.equal(no_of_scanlines_per_image, \
                                       transposed_input.shape[0]), \
                            [tf.constant("assert_transposed_0 Failed")], \
                            name="assert_transposed_0")

#The following Assert is not accepted by TensorFlow. The reason:
#    ValueError: Tried to convert 'y' to a tensor and failed. 
#    Error: Cannot convert an unknown Dimension to a Tensor: ?
#
#How is this different from the one preceding, and the one folowing?
#HYPOTHESIS: What is dimension 1, was, before permutation, dimension 0.
#           Dimension 0 in the placehlder has been marked as None. That is why,
            #I think, the error mesaage has the phrase "an unknown Dimension"
#
#assert_transposed_1 = tf.Assert(tf.equal(batch_size, \
#                                         transposed_input.shape[1]), \
#                            [tf.constant("assert_transposed_1 Failed")], \
#                            name="assert_transposed_1")

#The following also gives the same error message. It would mean that the operation
#tf.transpose is not being performed. If it were then _input_images would have been loaded -
# if my hypothesis was correct
#assert_transposed_1 = tf.Assert(tf.equal(batch_size, \
#                                 tf.transpose(_input_images, perm=[1, 0, 2]).shape[1]), \
#                            [tf.constant("assert_transposed_1 Failed")], \
#                            name="assert_transposed_1")
assert_transposed_2 = tf.Assert(tf.equal(no_of_elements_per_scanline, \
                                             transposed_input.shape[2]), \
                            [tf.constant("assert_transposed_2 Failed")], \
                            name="assert_transposed_2")

initial_hidden_state = tf.zeros([batch_size, \
                                 state_vector_dimensionality])

def rnn_step(previous_hidden_state, x):
        current_hidden_state = tf.tanh(
            tf.matmul(previous_hidden_state, Wh) +
            tf.matmul(x, Wx) + b_rnn)
        return current_hidden_state


# Getting all state vectors ***across time***
#This is the function that scans the images along the vertical axis
#line-by-line. The vertical axis is along dim = 0 in the transposed
#input. The scanning presents the sequence of vertical scans. The
#length of the sequence is therefore no_of_scanlines_per_image 
#(see assert below). For the selected element of the 
#scan sequence - the time-step -
#the applicable row of pixels, for the ***complete *** batch is 
#presented to rnn_step. 
#So at each point of the sequence of the vertical scan -
# - a time-step - the input is a matrix of dimension 
#[batch_size, no_of_elements_per_scanline]. Each row of the matrix has the
#pixels for the scan at a particular scan-step for the complete batch.
#SEE BELOW for assertions on all_hidden_states.
        
all_hidden_states = tf.scan(rnn_step,
                            transposed_input,
                            initializer=initial_hidden_state,
                            name='all_hidden_states') #'states')
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
# Weights for output layers
Wl = tf.Variable(tf.truncated_normal([state_vector_dimensionality, \
                                      no_of_classes],
                                     mean=0, stddev=.01))
bl = tf.Variable(tf.truncated_normal([no_of_classes],
                                     mean=0, stddev=.01))
#########################################################################
#This function needs to be sandwiched here.
        
# Apply linear layer to state vector
def get_linear_layer(hidden_state):

    return tf.matmul(hidden_state, Wl) + bl


#########################################################################
#    # Iterate across time, apply linear layer to all RNN outputs
all_outputs = tf.map_fn(get_linear_layer, \
                        all_hidden_states,\
                        name='FilterOPofAllHiddenStages')
# Get Last output -- h_28
output = all_outputs[-1]
##########################################################################
#A bunch of assertions to ilustrate my understanding
try:
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())

        batch_images, batch_labels = mnist.train.next_batch(batch_size)
        assert(2 == batch_images.ndim)
        assert(2 == batch_labels.ndim)
        assert('float32' == batch_images.dtype)
        assert(batch_size == batch_images.shape[0] and
               (no_of_scanlines_per_image * \
               no_of_elements_per_scanline) == batch_images.shape[1])
        assert(batch_size == batch_labels.shape[0] and
               no_of_classes == batch_labels.shape[1])
        
        #1st pixel (pixel 0), of 2nd row of scan (row 1),of the 1st image of the batch.
        image0_row1_pixel0 = batch_images[0, no_of_elements_per_scanline]
#        print("image0_row1_pixel0: ", image0_row1_pixel0)
        #Inserting a value at a pixel can help trace effects of shape transformations
        pixel_val = 1.0E+04
        batch_images[0, no_of_elements_per_scanline] = pixel_val
        assert(pixel_val == batch_images[0, no_of_elements_per_scanline])
        # Reshape data to get 28 sequences of 28 pixels per image
        batch_images = batch_images.reshape(batch_size, \
                                   no_of_scanlines_per_image, \
                                   no_of_elements_per_scanline)
        assert(batch_size == batch_images.shape[0] and
               no_of_scanlines_per_image == batch_images.shape[1] and
               no_of_elements_per_scanline == batch_images.shape[2])

        pixel_no = 0; scanline_no = 1
#        print("image0_row1_pixel0: ", batch_images[0, scanline_no, pixel_no])
        #This assertion shows how the rows and pixels are organized
        assert(pixel_val == batch_images[0, scanline_no, pixel_no])

        x = sess.run(transposed_input,
                              feed_dict={_input_images: batch_images, \
                                         labels_true: batch_labels})
        assert(no_of_scanlines_per_image == x.shape[0] and
               batch_size == x.shape[1] and
               no_of_elements_per_scanline == x.shape[2])

        sess.run(assert_transposed_0, \
                 feed_dict={_input_images: batch_images, \
                                              labels_true: batch_labels})
#        sess.run(assert_transposed_1, \
#                 feed_dict={_input_images: batch_images, \
#                                              labels_true: batch_labels})
#        sess.run(assert_transposed_2, \
#                 feed_dict={_input_images: batch_images, \
#                                              labels_true: batch_labels})

#        The transposed input - dimensions as asserted above - is 
#        tf.scan'ed and all_hidden_states built step-by-step. 
#        
#        At each step 
#            the hidden state at the previous step is multiplied by Wh. 
#            Wh has dimensions of [state_vector_dimensionality, state_vector_dimensionality])). 
#            
#            The first step uses a previous hidden state initialized as 
#            zeros[batch_size, state_vector_dimensionality]). The result is 
#            [batch_size, state_vector_dimensionality]
#            
#            The scanned input (from tf.scan) has dimension of 
#            [batch_size, no_of_elements_per_scanline]. This is multiplied 
#            by Wx which has a dimension
#            [no_of_elements_per_scanline, state_vector_dimensionality]
#            This also results in [batch_size, state_vector_dimensionality]
#            
#            The third term is the bias which is a row vector of
#            dimension [state_vector_dimensionality]. This single row is 
#            "broadcast" to all the batch_size rows of the other two terms
        x = sess.run(b_rnn,
                              feed_dict={_input_images: batch_images, \
                                         labels_true: batch_labels})
        assert(1 == np.asmatrix(x).shape[0] and
               state_vector_dimensionality == np.asmatrix(x).shape[1])

#        
#        The number of times that the input is scanned equals 
#        transposed_input[0] i.e. no_of_scanlines_per_image. 
#        Therefore, all_hidden_states[0] is no_of_scanlines_per_image 
#        as asserted below
 
        x = sess.run(all_hidden_states,
                              feed_dict={_input_images: batch_images, \
                                         labels_true: batch_labels})
        assert(no_of_scanlines_per_image == x.shape[0] and
               batch_size == x.shape[1] and
               state_vector_dimensionality == x.shape[2])
        
#        This checks the True branch of the tf.cond() node.
#        It returns the tensor, all_hidden_states, 
#        if the condition holds True
        x = sess.run(cond_T_all_hidden_states,
                              feed_dict={_input_images: batch_images, \
                                         labels_true: batch_labels})
        assert(no_of_scanlines_per_image == x.shape[0] and
               batch_size == x.shape[1] and
               state_vector_dimensionality == x.shape[2])
        
#        This checks the False branch of the tf.cond() node.
#        It returns the tensor transpose [1,0,2] of, all_hidden_states, 
#        if the condition fails
        x = sess.run(cond_F_all_hidden_states,
                              feed_dict={_input_images: batch_images, \
                                         labels_true: batch_labels})
        assert(batch_size == x.shape[0] and
               no_of_scanlines_per_image == x.shape[1] and
               state_vector_dimensionality == x.shape[2])
        
#        Each hidden layer (of all_hidden_states) is multiplied by 
#        Wl which has dimension of [state_vector_dimensionality, no_of_classes],
#        to obtain all_outputs which thus has the dimensions as 
#        asserted next
        
        x = sess.run(all_outputs,
                              feed_dict={_input_images: batch_images, \
                                         labels_true: batch_labels})
        assert(no_of_scanlines_per_image == x.shape[0] and
               batch_size == x.shape[1] and
               no_of_classes == x.shape[2])

        x = sess.run(output,
                              feed_dict={_input_images: batch_images, \
                                         labels_true: batch_labels})
        assert(batch_size == x.shape[0] and
               no_of_classes == x.shape[1])

#        Will running the graph nodes also start the process of
#        logging? No file writer object hass been created, as done
#        for the session below. Needs to be checked.
#        As a precaution clearing any cache, before going to the 
#        session where the logging is needed. 
#        tf.summary.FileWriterCache.clear()

#
##########################################################################
#The block of code above has been reading batches from the data file.
#Need to reset the counts so that training can process from the start of the file    
    
    mnist.train.reset_counts()

finally:
    #This is needed only when the file is run in spyder.
    #A re-run will cause an exception.
    #If run from the command line there is no problem
    tf.reset_default_graph()
    print("Exiting from finally in: ", __file__)
