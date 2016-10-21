import numpy as np

import data

def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def cross_entropy_loss(prob, y):
    return -y * np.log(prob) - (1 - y) * np.log(1 - prob)



def binlogreg_train(X, Y_, param_niter=100000, param_delta=0.1):
    """
    Method that trains binary logistic regresion for given data
    and returns parameters of trained model.

    Parameters
    ----------
    X : 2-D array_like, of shape (N, 2)
        data
    Y_ : 1-D array_like, of length N
        labels for passed data

    Returns
    -------
    w : 1-D array_like, of length 2
        array of weights for each feature
    b: scalar value
        bias parameter

    """
    n_samples = len(X)
    N = 2 # number of features
    w = np.random.randn(N)
    b = 0


    for i in range(param_niter):
        scores = np.dot(X, w) + b

        # vjerojatnosti razreda c_1
        probs = sigmoid(scores)
        loss  = np.sum(cross_entropy_loss(probs, Y_))

        if i % 10 == 0:
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
    '''
    Parameters
    ----------
    X : 2-D array_like, of shape (N, 2)
        data
    w : 1-D array_like, of length 2
        array of weights for each feature
    b: scalar value
        bias parameter

    Returns
    -------
    probs : 1-D array_like, of length N
        probability of class 1 for each sample in X
    '''
    return sigmoid(np.dot(X, w) + b)


if __name__=="__main__":
    np.random.seed(100)

    # get the training dataset
    X, Y_ = data.sample_gauss_2d(2, 100)

    # train the model
    w, b = binlogreg_train(X, Y_)


    # evaluate the model on the training dataset
    probs = binlogreg_classify(X, w,b)
    Y = np.array(probs > 0.5, dtype=int)

    # report performance
    accuracy, recall, precision = data.eval_perf_binary(Y, Y_)
    AP = data.eval_AP(Y_[probs.argsort()])
    print (accuracy, recall, precision, AP)
