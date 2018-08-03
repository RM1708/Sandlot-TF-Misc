#From: https://stackoverflow.com/questions/33633370/how-to-print-the-value-of-a-tensor-object-in-tensorflow

# Initialize session
import tensorflow as tf

# Some tensor we want to print the value of
a = tf.constant([1.0, 3.0])

# Add print operation
# From the doc:
#Note: This op prints to the standard error. 
#It is not currently compatible with jupyter notebook
#
#Run from command line to see the print output
c = tf.Print(a, [a, a, a], message="This is a: ")

# Add more elements of the graph using a
b = tf.add(a, c)

sess = tf.InteractiveSession()
print("\n#To see the side effect of tf.Print() run from the command line.\n")
print("From InteractiveSession. \nb: ", b.eval())
#This does not result in a message and print of the list of tensors
print("c.eval(): ", c.eval())
print()

sess.close()

with tf.Session() as sess:
    print("From Session manager \nb:",  sess.run(b))
    #This does not result in a message and print of the list of tensors
    print("c.eval(): ", sess.run(c))
        

