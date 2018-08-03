#From: https://stackoverflow.com/questions/37898478/is-there-a-way-to-get-tensorflow-tf-print-output-to-appear-in-jupyter-notebook-o?rq=1
# answered Sep 22 '16 at 22:07 Eric Czech

import tensorflow as tf
import numpy as np

def tf_print(tensor, transform=None):

    # Insert a custom python operation into the graph that does nothing but print a tensors value 
    def print_tensor(x):
        # x is typically a numpy array here so you could do anything you want with it,
        # but adding a transformation of some kind usually makes the output more digestible
        #
#        print(x if transform is None else transform(x))
        #The above is neat syntactic sugar
        if transform is None: 
            print("\tContents of np.array x: ",x)
        else:
           print("\tValues of x, after applying transformation: ", transform(x))

        return x
    log_op = tf.py_func(print_tensor, [tensor], [tensor.dtype])[0] #Wraps a python function and uses it as a TensorFlow op.
                                                    #See https://www.tensorflow.org/api_docs/python/tf/py_func
                                                    #Given a python function (in this case print_tensor()), 
                                                    #which takes ***numpy arrays as its arguments*** 
                                                    #and ***returns numpy arrays*** as its outputs.
                                                    #
                                                    #tf.py_func Returns A list of Tensor or a single Tensor which func computes
                                                    
    with tf.control_dependencies([log_op]): #https://www.tensorflow.org/api_docs/python/tf/control_dependencies
                                            # A list of Operation or Tensor objects which must be executed or 
                                            #computed before running the operations defined in the context
                                            #
                                            #When eager execution is enabled, any callable object in the 
                                            #control_inputs list will be called
                                            #A context manager that specifies control dependencies for all 
                                            #operations constructed within the contex
     
        pass
        #If the following op is outside the with block log_op will not be called, thus
        #print will not take place
        res = tf.identity(tensor)   #triggers evaluation of the dependenies

    with tf.Session() as sess:
        print("\nAlternative to activating via control_dependencies")
        print("NOTE: This prints before that via control_dependencies")
        print("And before the prints in the two sessions below!!!")
        sess.run(log_op)

    # Return the given tensor
    return res


# Now define a tensor and use the tf_print function much like the tf.identity function
#Creates a 2-D array of random numbers with mean = 100.0, sigma = 10 and seed = 0
tensor = tf_print(tf.random_normal([100, 100], mean=100.0, \
                                   stddev=10.0, seed=0), \
                  transform=lambda x: [np.min(x), np.max(x)])
#An identity node in the graph
tensor_1 = tf.identity(tensor)

# This will print the transformed version of the tensors actual value 
# (which was summarized to just the min and max for brevity)
with tf.Session() as sess: #= tf.InteractiveSession()
    print ("\nEdge: tensor from node tf_print:")
    res = sess.run([tensor])
    print("\tShape of Result of sess.run([tensor])", np.asarray(res).shape)

with tf.Session() as sess:
    print ("\nEdge:  tensor_1 from node tf.identity:")
    res = sess.run([tensor_1])
    print("\tShape of Result of sess.run([tensor])", np.asarray(res).shape)


