#From https://stackoverflow.com/questions/37898478/is-there-a-way-to-get-tensorflow-tf-print-output-to-appear-in-jupyter-notebook-o?rq=1

import tensorflow as tf

a = tf.constant(" RM")

b = tf.Print(a, [a], 'hi')

sess = tf.Session()

print("\nb: ", (b.eval(session=sess)).decode())

sess.close()


