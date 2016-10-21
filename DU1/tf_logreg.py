import tensorflow as tf

class TFLogreg:
    def __init__(self, D, C, param_delta=0.5, param_lambda=1e-4):
        """Arguments:
           - D: dimensions of each datapoint
           - C: number of classes
           - param_delta: training step
        """

        # data
        self.X  = tf.placeholder(tf.float32, [None, D])
        self.Yoh_ = tf.placeholder(tf.float32, [None, C])

        W = tf.Variable(tf.random_normal([C, D]))
        b = tf.Variable(tf.zeros([C]))


        # model
        scores = tf.matmul(self.X, W, transpose_b=True) + b
        self.probs = tf.nn.softmax(scores)


        # formulacija gubitka: self.loss
        log_prob = -tf.log(self.probs)
        err_loss = tf.reduce_sum(self.Yoh_ * log_prob, 1)
        reg_loss =  tf.nn.l2_loss(W)

        self.loss = tf.reduce_mean(err_loss) + param_lambda * reg_loss


        # optimisation
        trainer = tf.train.GradientDescentOptimizer(param_delta)
        self.train_step = trainer.minimize(self.loss)

        # instanciranje izvedbenog konteksta: self.session
        self.session = tf.Session()


    def train(self, X, Yoh_, param_niter):
        """Arguments:
           - X: actual datapoints [NxD]
           - Yoh_: one-hot encoded labels [NxC]
           - param_niter: number of iterations
        """
        self.session.run(tf.initialize_all_variables())
        data_dict = {self.X: X, self.Yoh_: Yoh_}

        # train loop
        for i in range(param_niter):
            val_loss, _ = self.session.run([self.loss, self.train_step], feed_dict=data_dict)
            if i % 100 == 0: print("{}\t{}".format(i, val_loss))

    def eval(self, X):
        """Arguments:
           - X: actual datapoints [NxD]
           Returns: predicted class probabilites [NxC]
        """
        probs =  self.session.run(self.probs, {self.X: X})
        return probs

if __name__ == "__main__":
    import numpy as np
    import data
    import matplotlib.pyplot as plt

    tf.reset_default_graph()
    np.random.seed(100)
    tf.set_random_seed(100)

    C = 3
    n = 100
    X, Y_, Yoh_ = data.sample_gauss_2d(C, n, one_hot=True)


    tflr = TFLogreg(X.shape[1], Yoh_.shape[1], 0.1, 0.25)
    tflr.train(X, Yoh_, 1000)

    probs = tflr.eval(X)
    Y = probs.argmax(axis=1)
    decfun = lambda x: tflr.eval(x).argmax(axis=1)


    # eval
    mat, classes = data.confusion_mat(y_pred=Y, y_true=Y_)
    APs = data.eval_AP_multi(Y_=Y_, probs=probs)
    print(mat)
    print(APs)

    # plot
    bbox=(np.min(X, axis=0), np.max(X, axis=0))

    data.graph_surface(decfun, bbox, offset=0.5)
    data.graph_data(X, Y_, Y)
    plt.show()

