import tensorflow as tf
import numpy as np

z = [x for x in range(10)]

as_2D_array = tf.reshape(z, [2, 5])

step = tf.group(as_2D_array)

sess = tf.InteractiveSession()

print(step.run())
print(sess.run(as_2D_array))
sess.close()

print(z)

print(np.shape(z))
print(np.asarray(z).shape)
print(np.asarray(z).reshape(1,-1))
print(np.asarray(z)[0])
print(np.asarray(z).reshape(1,-1).shape)
print(np.asarray(z).reshape(1,-1).shape)
print(np.asarray(z).reshape(1, 2, 5, 1).shape)
print(np.asarray(z).reshape(1, 2, 5, -1).shape)
print(np.asarray(z).reshape(1, 2, 5, 1))
print(np.asarray(z).reshape(1, 2, 5, 1)[0])
print(np.asarray(z).reshape(1, 2, 5, 1)[0, : , : , 0])
print(np.asarray(z).reshape(1, 2, 5, 1)[0].shape)
print(np.asarray(z).reshape(1, 2, 5, 1)[0, : , : , 0].shape)


