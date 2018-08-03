#From: https://stackoverflow.com/questions/44251328/tensorflow-print-all-placeholder-variable-names-from-meta-graph/44371483#44371483
import tensorflow as tf

import os
import errno

try:
    def save():
        v1 = tf.placeholder(tf.float32, name="v1") 
        v2 = tf.placeholder(tf.float32, name="v2")
        v3 = tf.multiply(v1, v2)
        vx = tf.Variable(10.0, name="vx")
        v4 = tf.add(v3, vx, name="v4")
        saver = tf.train.Saver()
        
        print("\n\tCollection Keys: \n\t", \
              tf.get_default_graph().get_all_collection_keys())
        for v in tf.get_default_graph().get_collection("variables"):
            print("\n\tvariables:\n\t",v)
        for v in tf.get_default_graph().get_collection("trainable_variables"):
            print("\n\ttrainable_variables: \n\t",v)

        sess = tf.Session()
        sess.run(tf.global_variables_initializer())
        sess.run(vx.assign(tf.add(vx, vx)))
        result = sess.run(v4, feed_dict={v1:12.0, v2:3.3})
        print("\n\t\"v1 = 12, v2 = 3.3, vx = 10. Result (v1*v2 + (vx + vx))\": ", \
              result)
        
        saver_file = "/home/rm/tmp/models/model_ex1"
        if not os.path.exists(os.path.dirname(saver_file)):
            try:
                os.makedirs(os.path.dirname(saver_file))
            except OSError as exc: # Guard against race condition
                if exc.errno != errno.EEXIST:
                    raise
        
        saver.save(sess, saver_file)
        sess.close()
        
    def restore():
        saver = tf.train.import_meta_graph("/home/rm/tmp/models/model_ex1.meta")

        print("\n\tCollection Keys: \n\t", tf.get_default_graph().get_all_collection_keys())
        for v in tf.get_default_graph().get_collection("variables"):
            print("\n\tvariables:\n\t",v)
        for v in tf.get_default_graph().get_collection("trainable_variables"):
            print("\n\ttrainable_variables: \n\t",v)

        sess = tf.Session()
        saver.restore(sess, "/home/rm/tmp/models/model_ex1")
        result = sess.run("v4:0", feed_dict={"v1:0": 12.0, "v2:0": 4.0})
        print("\n\t\"v1 = 12, v2 = 4, vx = 10. Result (v1*v2 + (vx + vx))\": ", result)
        
        sess.close()
    
    imported_graph = tf.get_default_graph()
    graph_op = imported_graph.get_operations()
    
    graph_op_file = "/home/rm/tmp/models/graph_op.txt"
    if not os.path.exists(os.path.dirname(graph_op_file)):
        try:
            os.makedirs(os.path.dirname(graph_op_file))
        except OSError as exc: # Guard against race condition
            if exc.errno != errno.EEXIST:
                raise
    
    with open(graph_op_file, 'w') as f:
        for i in graph_op:
            f.write(str(i))

    print("\nFIRST form Graph & Save:")
    save()
    
    print("\nNOW Restore:")
    restore()
    
    print("\n\tDONE: \n", __file__)

finally:
    tf.reset_default_graph()

