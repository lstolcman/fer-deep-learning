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


def _as_one_hot(C, Y):
    n = len(Y)
    Yoh = np.zeros((n, C))
    Yoh[range(n), Y] = 1
    return Yoh


def sample_gauss_2d(C, N, one_hot=False):
    Gs = [Random2DGaussian() for i in range(C)]
    Y = np.random.randint(0, C, N)
    X = np.array([Gs[c].get_sample()[0] for c in Y])
    if one_hot:
        return X, Y, _as_one_hot(C, Y)
    return X, Y


def sample_gmm_2d(K, C, N, one_hot=False):
    Gs = [Random2DGaussian() for i in range(K)]
    G_labels = np.random.randint(0, C, K)

    X = np.array([g.get_sample(N) for g in Gs]).reshape(-1, 2)
    Y = np.repeat(G_labels, N)
    if one_hot:
        return X, Y, _as_one_hot(C, Y)
    return X, Y


def graph_data(X, Y_, Y, special=[]):
    correct = Y_ == Y
    wrong = Y_ != Y
    colors = Y_
    size = np.array([20] * len(Y_))
    size[special] *= 3

    vmin = colors.min()
    vmax = colors.max()

    plt.scatter(X[correct, 0], X[correct, 1], marker='o', vmin=vmin, vmax = vmax,
                c=colors[correct], edgecolor='k', cmap='gray', s=size[correct])
    plt.scatter(X[wrong, 0], X[wrong, 1], marker='s', vmin=vmin, vmax = vmax,
                c=colors[wrong], edgecolor='k', cmap='gray', s=size[wrong])



def graph_surface(fun, rect, offset, width=250, height=250):
    (x_min, y_min), (x_max, y_max) = rect

    xx, yy = np.meshgrid(np.linspace(x_min, x_max, width), np.linspace(y_min, y_max, height))
    XX = np.c_[xx.ravel(), yy.ravel()]
    Z = fun(XX).reshape(xx.shape)

    delta = np.abs(Z-offset).max()
    plt.pcolormesh(xx, yy, Z, vmin=offset-delta, vmax=offset+delta)
    plt.contour(xx, yy, Z, levels=[offset])


# EVAL
def confusion_mat(y_true, y_pred):
    C = sorted(set(y_true) | set(y_pred))
    n_classes = len(C)
    mat = np.zeros((n_classes, n_classes), dtype='int')
    pairs = np.vstack((y_true, y_pred)).T

    for i, c_i in enumerate(C):
        for j, c_j in enumerate(C):
            mat[i][j] = (pairs == (c_i, c_j)).all(axis=1).sum()

    return mat, C


def get_stats(bin_conf_mat):
    tn, fp, fn, tp = bin_conf_mat.flatten()

    accuracy = (tp+tn) / (tn + tp + fn + fp)
    precision = tp / (tp + fp)
    recall = tp / (tp + fn
                  )
    return accuracy, recall, precision

def eval_perf_binary(Y, Y_):
    mat, classes = confusion_mat(Y_, Y)
    return get_stats(mat)


def twoway_confusion_matrix(mat, i):
    n_samples_i = mat[i].sum()
    n_samples = mat.sum()

    tp = mat[i, i]
    fp = mat[:, i].sum() - tp
    fn = n_samples_i - tp
    tn = n_samples - tp - fp - fn
    return np.array([[tp, fp], [fn, tn]])


def eval_perf_multi(Y, Y_):
    mat, classes = confusion_mat(Y_, Y)
    precision = []; recall = []

    for i in range(len(classes)):
        mat_i = twoway_confusion_matrix(mat, i)
        acc, p, r = get_stats(mat_i)
        precision.append(p); recall.append(r)


    precision = np.array(precision)
    recall = np.array(recall)
    accuracy = mat.trace() / mat.sum()
    return accuracy, mat, classes, precision, recall


def _precision(Y, i):
    clf_as_one = Y[i:]

    tp = (clf_as_one == 1).sum()
    fp = (clf_as_one == 0).sum()

    return tp / (tp + fp)

def eval_AP(Yr):
    Yr = np.array(Yr)
    n = len(Yr)

    return np.sum(_precision(Yr, i)*Yr[i] for i in range(n)) / np.sum(Yr)


def eval_AP_multi(Y_, probs):
    n, C = probs.shape
    return [eval_AP(Y_[probs[:, c].argsort()]) for c in range(C)]

