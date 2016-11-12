import tensorflow as tf
import os

class TFDeep:
    def __init__(self, config, param_delta, param_lambda=1e-4, activation=tf.nn.relu):
        """Arguments:
           - D: dimensions of each datapoint
           - C: number of classes
           - param_delta: training step
        """

        D = config[0]; C = config[-1]
        n_layers = len(config[1:])

        # data
        self.X  = tf.placeholder(tf.float32, [None, D])
        self.Yoh_ = tf.placeholder(tf.float32, [None, C])

        activations = [activation] * (n_layers-1) + [tf.nn.softmax]


        reg_loss = 0
        prev_out = self.X
        for i, (prev, next) in enumerate(zip(config, config[1:])):
            Ws = tf.Variable(tf.random_normal([next, prev]), name='W%s' % i)
            bs = tf.Variable(tf.random_normal([next]), name='b%s' % i)

            s = tf.add(tf.matmul(prev_out, Ws, transpose_b=True), bs)
            prev_out = activations[i](s)

            reg_loss += tf.nn.l2_loss(Ws)

        # output
        self.probs = prev_out

        err_loss = tf.reduce_mean(-tf.reduce_sum(self.Yoh_ * tf.log(self.probs+1e-10), 1))
        self.loss = err_loss + param_lambda * reg_loss

        self.train_step = tf.train.GradientDescentOptimizer(param_delta).minimize(self.loss)
        self.session = tf.Session()



    def train(self, X, Yoh_, param_niter, verbose=True):
        """Arguments:
           - X: actual datapoints [NxD]
           - Yoh_: one-hot encoded labels [NxC]
           - param_niter: number of iterations
        """
        # incijalizacija parametara
        self.session.run(tf.initialize_all_variables())
        data_dict = {self.X: X, self.Yoh_: Yoh_}

        # train loop
        for i in range(param_niter):
            val_loss, _ = self.session.run([self.loss, self.train_step], feed_dict=data_dict)
            if verbose and i % 100 == 0: print("{}\t{}".format(i, val_loss));

    def eval(self, X):
        """Arguments:
           - X: actual datapoints [NxD]
           Returns: predicted class probabilites [NxC]
        """
        probs =  self.session.run(self.probs, {self.X: X})
        return probs


if __name__ == '__main__':
    import numpy as np
    import data
    import matplotlib.pyplot as plt

    tf.reset_default_graph()
    tf.set_random_seed(130)
    np.random.seed(42)


    X, Y_, Yoh_ = data.sample_gmm_2d(4, 2, 10, one_hot=True)

    # izgradi graf:
    config = [X.shape[1], 10, 10, Yoh_.shape[1]]
    nn = TFDeep(config, 0.05, 1e-4, tf.nn.sigmoid )
    nn.train(X, Yoh_, 4000)

    probs = nn.eval(X)
    Y = probs.argmax(axis=1)


    decfun = lambda x: nn.eval(x)[:,1]
    bbox=(np.min(X, axis=0), np.max(X, axis=0))
    data.graph_surface(decfun, bbox, offset=0.5)
    data.graph_data(X, Y_, Y)
    plt.show()
