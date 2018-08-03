import tensorflow as tf
import numpy as np
from pprint import pprint as pp

z = [x for x in range(10)]

as_2D_array = tf.reshape(z, [2, 5])

step = tf.group(as_2D_array)

sess = tf.InteractiveSession()

print("\nstep.run(): ", sess.run(step))
print("sess.run(as_2D_array): \n", sess.run(as_2D_array), "\n")
sess.close()

print("z: \n", z, "\n")

print(np.shape(z))
print(np.asarray(z).shape)
print(np.asarray(z).reshape(1,-1))
print("np.asarray(z)[0]: ", np.asarray(z)[0])
print()
print(np.asarray(z).reshape(1, 2, 5, 1).shape)
print()
print("reshape(1, 2, 5, 1): \n", \
      (np.asarray(z).reshape(1, 2, 5, 1)))

print("\nreshape(1, 2, 5, 1)[0]: \n", \
      np.asarray(z).reshape(1, 2, 5, 1)[0])
print("reshape(1, 2, 5, 1)[0].shape: \n", \
      np.asarray(z).reshape(1, 2, 5, 1)[0].shape)
print()
print("reshape(1, 2, 5, 1)[0, : , : , 0]: \n", \
      np.asarray(z).reshape(1, 2, 5, 1)[0, : , : , 0])
print("reshape(1, 2, 5, 1)[0, : , : , 0].shape: \n", \
      np.asarray(z).reshape(1, 2, 5, 1)[0, : , : , 0].shape)

"""
reshape(1, 2, 5, 1): 
 [
  [<<<<<<<<< 1 element at this level
   [ <<<<<<< 2 elements at this
    [ <<<<<< 5 elements at this
     0 <<<<< 1 element at this
    ]
    [
     1
    ]
    [
     2
    ]
    [
     3
    ]
    [
     4
    ]
   ]

   [
    [5]
    [6]
    [7]
    [8]
    [9]
   ]
  ]
 ]
"""
