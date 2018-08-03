
#From: https://stackoverflow.com/questions/49735217/tensorflow-leaks-1280-bytes-with-each-session-opened-and-closed
import tensorflow as tf
#tensorflow/contrib/memory_stats/python/ops/memory_stats_ops.py.
import sys
n_Iterations=int(sys.argv[1])
def open_and_close_session():
   with tf.Session() as sess:
      pass
for _ in range(n_Iterations):
   open_and_close_session()
with tf.Session() as sess:
   print("bytes used=",sess.run(tf.contrib.memory_stats.BytesInUse())
#tf.contrib.memory_stats.BytesInUse()