# py2
from __future__ import division

import random
import numpy as np
import matplotlib.pyplot as plt


def _sample_uniform(lo, hi):
    return np.random.random_sample()* (hi - lo) + lo

def _rotation_matrix2D(angle):
    theta = np.radians(angle)
    c, s = np.cos(theta), np.sin(theta)
    return np.array([[c, -s], [s, c]])

class Random2DGaussian:

    def __init__(self, minx=0, maxx=10, miny=0, maxy=10):

        mu = _sample_uniform(minx, maxx), _sample_uniform(miny, maxy)
        self.mu = np.array(mu)


        eigvalx = (np.random.random_sample()*(maxx - minx)/5)**2
        eigvaly = (np.random.random_sample()*(maxy - miny)/5)**2
        D = np.diag([eigvalx, eigvaly])

        R = _rotation_matrix2D(np.random.random_sample() * 360)
        self.sigma = np.dot(R.T, np.dot(D, R))

    def get_sample(self, n_samples=1):
        return np.random.multivariate_normal(self.mu, self.sigma, n_samples)


# def sample_gauss_2d(C, N):
#     Gs = [Random2DGaussian() for i in range(C)]
#     Y = np.random.randint(0, C, N)
#     X = np.array([Gs[c].get_sample()[0] for c in Y])
#     return X, Y

def sample_gauss_2d(C, N):
    X = []
    y = []
    for i in range(C):
        X.extend(Random2DGaussian().get_sample(N))
        y.extend([i]*N)

    return np.array(X), np.array(y)



def confusion_mat(y_true, y_pred):
    C = sorted(set(y_true) | set(y_pred))
    n_classes = len(C)
    mat = np.zeros((n_classes, n_classes), dtype='int')
    pairs = np.vstack((y_true, y_pred)).T

    for i, c_i in enumerate(C):
        for j, c_j in enumerate(C):
            mat[i][j] = (pairs == (c_i, c_j)).all(axis=1).sum()

    return mat, C


def eval_perf_binary(Y, Y_):
    mat, classes = confusion_mat(Y_, Y)
    tn, fp, fn, tp = mat.flatten()

    accuracy = (tp+tn) / (tn + tp + fn + fp)
    precision = tp / (tp + fp)
    recall = tp / (tp + fn
                  )
    return accuracy, recall, precision


def _precision(Y, i):
    clf_as_one = Y[i:]

    tp = (clf_as_one == 1).sum()
    fp = (clf_as_one == 0).sum()

    return tp / (tp + fp)

def eval_AP(Yr):
    Yr = np.array(Yr)
    n = len(Yr)

    return np.sum(_precision(Yr, i)*Yr[i] for i in range(n)) / np.sum(Yr)



def graph_data(X, Y_, Y):
    """
        Parameters
    ----------
    X : 2-D array_like, of shape (N, 2)
        data
    Y_ : 1-D array_like, of length N
        true classes
    Y: scalar value
        predicted classes
    """

    pairs = np.vstack((Y_, Y)).T
    find = lambda x: (pairs == x).all(axis=1)

    tp = find((1,1))
    tn = find((0,0))
    fp = find((0, 1))
    fn = find((1, 0))


    tp = plt.scatter(X[tp, 0], X[tp, 1], marker='o', color='gray', edgecolor='k')
    tn = plt.scatter(X[tn, 0], X[tn, 1], marker='o', color='w', edgecolor='k')

    fn = plt.scatter(X[fn, 0], X[fn, 1], marker='s', color='gray', edgecolor='k')
    fp = plt.scatter(X[fp, 0], X[fp, 1], marker='s', color='w', edgecolor='k')


    plt.legend((tp, tn, fp, fn), ('tp', 'tn', 'fp', 'fn'),
           loc='best', ncol=2, fontsize=8)


if __name__=="__main__":
    G = Random2DGaussian()
    X = G.get_sample(100)
    plt.scatter(X[:,0], X[:,1])

    plt.title('mu %s\nsigma %s' % (G.mu, G.sigma))
    plt.show()



    def myDummyDecision(X):
        scores = X[:,0] + X[:,1] - 5
        return scores

    np.random.seed(100)

    # get the training dataset
    X,Y_ = sample_gauss_2d(2, 100)

    # get the class predictions
    Y = myDummyDecision(X)>0.5

    # graph the data points
    graph_data(X, Y_, Y)


    # show the results
    plt.show()
