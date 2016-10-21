# Tensorflow demo
import tensorflow as tf
tf.reset_default_graph()
a = tf.constant(5)
b = tf.constant(8)
x = tf.placeholder(dtype='int32')
c = a + b * x
d = b * x


session = tf.Session()
c_val = session.run(c, feed_dict={x: 5})
print(c_val)


import numpy as np

X = tf.placeholder(tf.float32, [2, 2])
Y = 3 * X + 5
z = Y[0,0]


sess = tf.Session()
Y_val = sess.run(Y, feed_dict={X: [[0,1],[2,3]]})
z_val = sess.run(z, feed_dict={X: np.ones((2,2))})

print(sess.run(tf.nn.l2_loss(Y), feed_dict={X: [[0,1],[2,3]]}))
print(Y_val, type(Y_val))
print(z_val, type(z_val))
print((Y_val * Y_val).sum())
