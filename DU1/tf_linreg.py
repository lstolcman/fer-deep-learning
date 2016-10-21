import numpy as np
from numpy.random import normal
from scipy import stats as stats
import tensorflow as tf
import matplotlib.pyplot as plt

f = lambda x: x + 3

np.random.seed(42)
Xs = np.random.uniform(-10, 10, 50)
Ys = f(Xs) + normal(0, 5e-2, len(Xs))
Xs, Ys = Xs.reshape(-1, 1), Ys.reshape(-1, 1)
n_samples = len(Xs)


X  = tf.placeholder(tf.float32, [None, 1])
Y_ = tf.placeholder(tf.float32, [None, 1])
a = tf.Variable(0.0)
b = tf.Variable(0.0)


Y = a*X + b
loss  = (1./(2*n_samples)) * (Y-Y_)**2
#tf.reduce_sum(tf.pow(Y - Y_, 2))/(2*n_samples)


trainer = tf.train.GradientDescentOptimizer(0.01)
train_op = trainer.minimize(loss)
grads_and_vars = trainer.compute_gradients(loss, [a, b])
optimize = trainer.apply_gradients(grads_and_vars)
grads_and_vars = tf.Print(grads_and_vars, [grads_and_vars], 'Status:')

grad_a = (1/n_samples) * tf.matmul(Y-Y_,  X, transpose_a=True)
grad_b = (1/n_samples) * tf.reduce_sum(Y-Y_)


with tf.Session() as sess:
    sess.run(tf.initialize_all_variables())
    plt.scatter(Xs, Ys, marker='o')

    for i in range(1000):
        val_loss, val_grads, da, db = sess.run([loss, grads_and_vars, grad_a, grad_b], feed_dict={X: Xs, Y_: Ys})
        sess.run(train_op, feed_dict={X: Xs, Y_: Ys})
        val_a, val_b= sess.run([a, b], feed_dict={X: Xs, Y_: Ys})

        if i% 100 == 0:
            print(val_a, val_b, val_loss.sum())
            print(val_grads)
            print(da, db)
            print()

    plt.plot(Xs, val_a*Xs + val_b, '-')
    plt.show()
