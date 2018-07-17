# -*- coding: utf-8 -*-
"""
Created on Thu Dec 29 00:39:23 2016

@author: tomhope
"""

import numpy as np
import tensorflow as tf

import argparse
import sys
from tensorflow.python import debug as tf_debug


WORDS_IN_A_SENTENCE = 6

batch_size = 5 #128
embedding_space_dimensionality = 7 #64
num_classes = 2
hidden_layer_size = 8   #32
times_steps = WORDS_IN_A_SENTENCE #6 #TODO: Is this connected to WORDS_IN_A_SENTENCE?
element_size = 1

##############################################################
#GENERATE simulated text sentences
#digit_to_word_map = {1: "One", 2: "Two", 3: "Three", 4: "Four", 5: "Five",
#                     6: "Six", 7: "Seven", 8: "Eight", 9: "Nine"}
#digit_to_word_map[0] = "PAD"

NUM_OF_SENTENCES = 10 #20000
MIN_LEN_OF_SENTENCE = 3
MAX_LEN_OF_SENTENCE = WORDS_IN_A_SENTENCE   #6
MIN_ODD_NUM = 1
MIN_EVEN_NUM = 2

def main(_):
    try:
        
        #################################################################
        #Construct TensorFlow Graph
            
        _inputs = tf.placeholder(tf.int32, shape=[batch_size, times_steps])

        number_of_distinct_words_found = 5
        with tf.name_scope("embeddings"):
            embeddings = tf.Variable(
                tf.random_uniform([number_of_distinct_words_found,
                                   embedding_space_dimensionality],
                                  -1.0, 1.0), name='embedding')
            embed = tf.nn.embedding_lookup(embeddings, _inputs)
        
        
        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())
            if FLAGS.debug:
                sess = tf_debug.LocalCLIDebugWrapperSession(sess, \
                                                          ui_type=FLAGS.ui_type)
            #PROBE DATA    
            x_batch = np.asarray(\
                        [[0] * 6, 
                         [1] * 6, 
                         [2] * 6, 
                         [3] * 6, 
                         [4] * 6])
            
            word_embeddings, Mbdngs = sess.run([embed, embeddings], \
                                               feed_dict={_inputs: x_batch})
            assert(batch_size == len(word_embeddings))
            assert(WORDS_IN_A_SENTENCE == len(word_embeddings[0]))
            assert(embedding_space_dimensionality == len(word_embeddings[0][0]))
            assert((batch_size, \
                    WORDS_IN_A_SENTENCE, \
                    embedding_space_dimensionality) == word_embeddings.shape)

            assert((batch_size, embedding_space_dimensionality) == \
                   np.asarray(Mbdngs).shape)

            print("\n\tDONE: ", __name__)
    finally:
        tf.reset_default_graph()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.register("type", "bool", lambda v: v.lower() == "true")
    parser.add_argument(
    "--ui_type",
    type=str,
    default="curses",
    help="Command-line user interface type (curses | readline)")
    parser.add_argument(
    "--debug",
    type="bool",
    nargs="?",
    const=True,
    default=False,
    help="Use debugger to track down bad values during training. "
    "Mutually exclusive with the --tensorboard_debug_address flag.")
    FLAGS, unparsed = parser.parse_known_args()
    tf.app.run(main=main, argv=[sys.argv[0]] + unparsed)
      