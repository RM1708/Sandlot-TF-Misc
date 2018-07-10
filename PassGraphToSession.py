# coding: utf-8
#From https://stackoverflow.com/questions/40558238/tensorflow-session-run-graph-defined-outside-under-tf-graph
#  answered Nov 12 '16 at 0:30 mrry

import tensorflow as tf

# define graph
my_graph = tf.Graph()

with my_graph.as_default():
    a = tf.constant(100., tf.float32, name='a')
    
with tf.Session(graph=my_graph) as sess:
    a = sess.graph.get_operation_by_name('a')
    print(sess.run(a))  # prints None
    
with tf.Session(graph=my_graph) as sess:
    a = sess.graph.get_operation_by_name('a').outputs[0]
    # Or you could do:
    # a = sess.graph.get_tensor_by_name('a:0')
    print(sess.run(a))  # prints '100.'
    
with tf.Session(graph=my_graph) as sess:
    a = sess.graph.get_tensor_by_name('a:0')
    print(sess.run(a))  # prints '100.'
