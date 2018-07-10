#From: https://stackoverflow.com/questions/33633370/how-to-print-the-value-of-a-tensor-object-in-tensorflow

# Initialize session
import tensorflow as tf

# Some tensor we want to print the value of
a = tf.constant([1.0, 3.0])

# Add print operation
a = tf.Print(a, [a], message="This is a: ")

# Add more elements of the graph using a
b = tf.add(a, a)

sess = tf.InteractiveSession()
print("From InteractiveSession. b: ", b.eval())
print()

sess.close()

with tf.Session() as sess:
        print("From Session manager b:",  sess.run(b))
        

