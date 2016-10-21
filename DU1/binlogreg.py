import numpy as np

def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def cross_entropy_loss(prob, y):
    return -y * np.log(prob) - (1 - y) * np.log(1 - prob)


def binlogreg_train(X, Y_, param_niter=100000, param_delta=0.1, verbose=False):
    n_samples = len(X)
    N = 2 # number of features
    w = np.random.randn(N)
    b = 0


    for i in range(param_niter):
        scores = np.dot(X, w) + b

        # vjerojatnosti razreda c_1
        probs = sigmoid(scores)
        loss  = np.sum(cross_entropy_loss(probs, Y_))

        if verbose and i % 10 == 0:
            print("iteration {}: loss {}".format(i, loss))

        # derivacije gubitka po klasifikacijskom rezultatu
        dL_dscores = probs - Y_

        # gradijenti parametara
        grad_w = 1.0/n_samples  * np.dot(dL_dscores, X)
        grad_b = 1.0/n_samples  * np.sum(dL_dscores)

        # poboljsani parametri
        w += -param_delta * grad_w
        b += -param_delta * grad_b

    return w, b

def binlogreg_classify(X, w, b):
    return sigmoid(np.dot(X, w) + b)
