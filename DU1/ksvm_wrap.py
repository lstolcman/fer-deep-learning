from sklearn.svm import SVC

class SVMWrapper:

    def __init__(self, X, Y_, c=1, g='auto'):
        self.model = SVC(gamma=g, C=c, probability=True)
        self.model.fit(X, Y_)

    def predict(self, X):
        return self.model.predict(X)

    def get_scores(self, X):
        return self.model.predict_proba(X)

    def support(self):
        return self.model.support_

if __name__ == '__main__':
    import numpy as np
    import data
    import matplotlib.pyplot as plt

    np.random.seed(100)

    C = 2
    n = 10
    X, Y_, Yoh_ = data.sample_gmm_2d(6, 2, 20, one_hot=True)


    model = SVMWrapper(X, Y_, c=1, g='auto')
    decfun = lambda x: model.get_scores(x)[:,1]
    probs = model.get_scores(X)
    Y = probs.argmax(axis=1)


    bbox=(np.min(X, axis=0), np.max(X, axis=0))
    data.graph_surface(decfun, bbox, offset=0.5)
    data.graph_data(X, Y_, Y, model.support())
    plt.show()

