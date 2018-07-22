
"""
FROM: https://stackoverflow.com/questions/48909330/meaning-of-values-returned-by-tensorflow-accuracy-metric

output:

accuracy_score 1.0
TF accuracy: (0.0, 1.0)
My accuracy: 1.0

accuracy_score 0.8
TF accuracy: (0.0, 0.8)
My accuracy: 0.8

accuracy_score 0.6
TF accuracy: (0.0, 0.6)
My accuracy: 0.6

accuracy_score 0.4
TF accuracy: (0.0, 0.4)
My accuracy: 0.4


"""
import tensorflow as tf
#from sklearn.metrics import accuracy_score

# true and predicted tensors
y_p = tf.placeholder(dtype=tf.int64)
y_t = tf.placeholder(dtype=tf.int64)

# Count true positives, true negatives, false positives and false negatives.
tp = tf.count_nonzero(y_p * y_t)
tn = tf.count_nonzero((y_p - 1) * (y_t - 1))
fp = tf.count_nonzero(y_p * (y_t - 1))
fn = tf.count_nonzero((y_p - 1) * y_t)

acc = tf.metrics.accuracy(predictions=y_p, labels=y_t)

# Calculate accuracy, precision, recall and F1 score.
accuracy = (tp + tn) / (tp + fp + fn + tn)

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    sess.run(tf.local_variables_initializer())
    for i in range(4):
#        sess.run(tf.global_variables_initializer())
        sess.run(tf.local_variables_initializer())


        if i == 0:
            yop = [0,0,0,0,0,0,0,0,0,0]
        elif i == 1:
            yop = [0,0,0,0,0,0,0,0,1,1]
        elif i == 2:
            yop = [1,1,1,0,0,0,0,0,0,1]
        else:
            yop = [0,1,1,1,1,1,1,0,0,0]
#        print('accuracy_score', accuracy_score([0,0,0,0,0,0,0,0,0,0], yop))
        tf_a = sess.run(acc, feed_dict={y_p: [0,0,0,0,0,0,0,0,0,0], y_t: yop})
        my_a = sess.run(accuracy, feed_dict={y_p: [0,0,0,0,0,0,0,0,0,0], y_t: yop})
        print("TF accuracy: {0}".format(tf_a))
        print("My accuracy: {0}".format(my_a))
        print()


