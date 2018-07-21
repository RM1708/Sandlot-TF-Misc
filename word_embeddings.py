# -*- coding: utf-8 -*-

"""
Th code here has been snipped from LSTM_supervised_embeddings.py.

The idea here is to check what exactly is it that embedding does.
Embedding is acheived by the two functions:
            word_embeddings = tf.Variable(
                tf.random_uniform([number_of_distinct_words_found,
                                   embedding_space_dimensionality],
                                  -1.0, 1.0), name='word_embeddings')
            
            embedded_word_ids = tf.nn.embedding_lookup(word_embeddings, \
            word_ids)

These two constitute the object under test

TEST DATA
The data has the same characteristic as the data used in
LSTM_supervised_embeddings.py, except for the following specialization:
    1. Only 5 sentences are used.Therefore, batch_size is 5
    2. A sentence has the same word-id repeated for the specified
    length of a sentence -six.
    3.Each sentence is composed from word-id unique to it. 
    Therefore there are just 5 unique word-ids.

SETUP
    1. Setup an array with the input data described above
    2. Add code to capture & print out the values returned 
    in the  arrays word_embeddings and embedded_word_ids
    1. Set embedding_space_dimensionality to 1

EXPT
    Run the code and observe 
        1. The values in the array word_embeddings
        2. The values returned in embedded_word_ids
    
    Repeat for different combination of values of
        1. embedding_space_dimensionality
        2. batch_size

OBSERVATIONS
    1. word_embeddings is a matrix with shape(number_of_distinct_words_found,
                                              embedding_space_dimensionality) 
    No surprise here; that is how word_embeddings is constructed
    2. For each unique word-id, word_embeddings has a vector.
    3. For each word-id in a sentence, the corresponding vector in embeddings 
    is substituted. To get the embedded word-ids for a given sentence:
        1. Get the index of the sentence in the input batch.
        2. Access the embedded_word_ids using that index for the first dimension.
        3. The matrix returned has rows which give the embeddings for each word 
        in the sentence. The rows are in the same order as the word-ids occur 
        in the sentence.
        4. It can be observed that all rows are identical. 
    4. For the batch of sentences given as input, the returned embedded_word_ids 
    has the shape (batch_size, 
                 num_of_word_ids_per_sentence, 
                 embedding_space_dimensionality)
    5. The shape can be asserted in the code.
    
CONCLUSION
    1. word-ids are integers. They are 0-D tensors. So how come using an
    embedding_space_dimensionality >> 1 will lead to denser vector space?
    2. The word_embeddings are mappings from word-ids (integers) to a vector of numbers 
    that generated from a uniform distribution in the range of [-1, 1]. This 
    DOES NOT GUARANTEE that the mappings are unique.
    3. Uniqueness of word_embeddings must be asserted.
    
"""

import numpy as np
import tensorflow as tf

import argparse
import sys
from tensorflow.python import debug as tf_debug


WORDS_IN_A_SENTENCE = 6

batch_size = 5 #128
embedding_space_dimensionality = 1 #3 #64
#With 3:
#embeddings:
# [[-0.7142718  -0.47191215 -0.5936341 ]
# [-0.5695057   0.56034803 -0.4264474 ]
# [-0.3007524  -0.74718165 -0.7662387 ]
# [-0.65923715 -0.31191254  0.9726834 ]
# [-0.47946024  0.49407053  0.9302335 ]]

#With1
#embeddings:
# [[-0.4371798 ]
# [-0.605181  ]
# [-0.7778797 ]
# [ 0.40726256]
# [ 0.8076551 ]]
#
#NOTE: Only 5 rows as there are only 5 distinct words that were fed.
 
num_of_word_ids_per_sentence = WORDS_IN_A_SENTENCE #6 

def main(_):
    try:
        #################################################################
        #Construct TensorFlow Graph
            
        word_ids = tf.placeholder(tf.int32, shape=[batch_size, \
                                                   num_of_word_ids_per_sentence])

        embeddings_3point = \
            np.asarray([
                        [0.5] * embedding_space_dimensionality,
                        [0.25] * embedding_space_dimensionality,
                        [0.75] * embedding_space_dimensionality,
                        [0.25] * embedding_space_dimensionality,
                        [0.75] * embedding_space_dimensionality,
                        ]) #The odd ids are mapped to 0.25. 
                            #Even ids are mapped to 0.75
                            #PAD is mapped to 0.5
                            #Mapping done only for word-ids 0, 1, 2, 3, 4
                            #Since only these are used in the test

#                        [0.25] * embedding_space_dimensionality,
#                        [0.75] * embedding_space_dimensionality,
#                        [0.25] * embedding_space_dimensionality,
#                        [0.75] * embedding_space_dimensionality,
#                        [0.75] * embedding_space_dimensionality,
                        
        number_of_distinct_words_found = 5
        with tf.name_scope("word_embeddings"):
            #Compare with:
            #https://www.tensorflow.org/guide/embedding#embeddings_in_tensorflow
#            word_embeddings = tf.Variable(
#                tf.random_uniform([number_of_distinct_words_found,
#                                   embedding_space_dimensionality],
#                                  -1.0, 1.0), name='word_embeddings')
            
            word_embeddings =tf.Variable(embeddings_3point)

            embedded_word_ids = tf.nn.embedding_lookup(word_embeddings, word_ids)

#The case dimensionality of 1 gives the following embedded_word_ids for
# the 6 words of each of the 5 sentence:
#
#1st sentence has 6 words all of id 0. Check these against embeddings[0] above
# [[[-0.4371798 ]
#  [-0.4371798 ]
#  [-0.4371798 ]
#  [-0.4371798 ]
#  [-0.4371798 ]
#  [-0.4371798 ]]
#
#2nd sentence has 6 words all with id 1. Check these against embeddings[1] above
# [[-0.605181  ]
#  [-0.605181  ]
#  [-0.605181  ]
#  [-0.605181  ]
#  [-0.605181  ]
#  [-0.605181  ]]
#3rd Sentence ...
#...        
#4th Sentence ...
#...        
#5th Sentence ...
#...        

        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())
            if FLAGS.debug:
                sess = tf_debug.LocalCLIDebugWrapperSession(sess, \
                                                          ui_type=FLAGS.ui_type)
            #PROBE DATA 
            #5 sentences.Each sentence has 6 words. 
            #All words in a sentence are the same.
            #Thus, the vocabulary used has just 5 words
            #The sentences have different words.
            #Each word is represented by a numeric id.
            #The ids are: 0, 1, 2, 3, 4
            five_sentences_with_6_word_ids = \
            np.asarray(\
                        [[0] * WORDS_IN_A_SENTENCE, 
                         [1] * WORDS_IN_A_SENTENCE, 
                         [2] * WORDS_IN_A_SENTENCE, 
                         [3] * WORDS_IN_A_SENTENCE, 
                         [4] * WORDS_IN_A_SENTENCE])
            
            embedded_word_ids, Mbdngs = sess.run([embedded_word_ids, word_embeddings], \
                                               feed_dict={word_ids: five_sentences_with_6_word_ids})

#            assert(batch_size == len(embedded_word_ids))
#            assert(WORDS_IN_A_SENTENCE == len(embedded_word_ids[0]))
#            assert(embedding_space_dimensionality == len(embedded_word_ids[0][0]))
            assert((batch_size, \
                    WORDS_IN_A_SENTENCE, \
                    embedding_space_dimensionality) == embedded_word_ids.shape)

            assert((batch_size, embedding_space_dimensionality) == \
                   np.asarray(Mbdngs).shape)
            
            print("\nembedded_word_ids:\n", embedded_word_ids)
            print("\nembeddings:\n", Mbdngs)

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
      