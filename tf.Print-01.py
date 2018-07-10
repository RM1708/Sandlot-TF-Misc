#From https://stackoverflow.com/questions/37898478/is-there-a-way-to-get-tensorflow-tf-print-output-to-appear-in-jupyter-notebook-o?rq=1

import tensorflow as tf

a = tf.constant(1.0)

a = tf.Print(a, [a], 'hi')

sess = tf.Session()

a.eval(session=sess)

sess.close()


