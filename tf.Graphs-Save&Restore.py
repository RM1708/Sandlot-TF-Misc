#From: https://stackoverflow.com/questions/44251328/tensorflow-print-all-placeholder-variable-names-from-meta-graph/44371483#44371483
import tensorflow as tf

def save():
    v1 = tf.placeholder(tf.float32, name="v1") 
    v2 = tf.placeholder(tf.float32, name="v2")
    v3 = tf.multiply(v1, v2)
    vx = tf.Variable(10.0, name="vx")
    v4 = tf.add(v3, vx, name="v4")
    saver = tf.train.Saver()
    sess = tf.Session()
    sess.run(tf.initialize_all_variables())
    sess.run(vx.assign(tf.add(vx, vx)))
    result = sess.run(v4, feed_dict={v1:12.0, v2:3.3})
    print(result)
    saver.save(sess, "./model_ex1")
    
def restore():
    saver = tf.train.import_meta_graph("./model_ex1.meta")
    print(tf.get_default_graph().get_all_collection_keys())
    for v in tf.get_default_graph().get_collection("variables"):
        print(v)
    for v in tf.get_default_graph().get_collection("trainable_variables"):
        print(v)
    sess = tf.Session()
    saver.restore(sess, "./model_ex1")
    result = sess.run("v4:0", feed_dict={"v1:0": 12.0, "v2:0": 4.0})
    print(result)



saver = tf.train.import_meta_graph('some_path/model.ckpt.meta')
imported_graph = tf.get_default_graph()
graph_op = imported_graph.get_operations()
with open('output.txt', 'w') as f:
    for i in graph_op:
        f.write(str(i))

