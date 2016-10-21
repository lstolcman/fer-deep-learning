import numpy as np

class fcann2:
    def __init__(self, H=5):
        self.H = H
        pass

    def fcann2_train(self, X, Y_, param_niter=100000, param_delta=0.1, verbose=False, step=100):
        C = np.max(Y_) + 1
        N, n_features = X.shape
        H = self.H
        all_rows = range(N)

        param_delta /= N
        # params
        W1 = np.random.randn(H, n_features)
        b1 = np.zeros(H)
        W2 = np.random.randn(C, H)
        b2 = np.zeros(C)

        for i in range(param_niter):
            S1 = np.dot(X, W1.T) + b1  # N x H
            H1 = np.maximum(0, S1)     # N X H

            S2 = np.dot(H1, W2.T) + b2   # N x C
            expscores = np.exp(S2)      # N x C
            sumexp = expscores.sum(axis=1) # N x 1

            probs = expscores / sumexp.reshape(-1,1)     # N x C
            correct_class_prob = probs[range(len(X)), Y_]
            correct_class_logprobs = -np.log(correct_class_prob)   # N x 1
            loss  = correct_class_logprobs.sum()

            if verbose and i % step == 0:
                print("iteration {}: loss {}".format(i, loss))

            dS2 = probs   # N x C
            dS2[all_rows,Y_] -= 1


            dW2 = np.dot(dS2.T, H1) # C x H
            db2 = dS2.sum(axis=0) # C x 1

            dH1 = np.dot(dS2, W2)  # N x H

            dS1 =  dH1  # N x H
            dS1[S1 <= 0] = 0

            dW1 = np.dot(dS1.T, X) # H x D
            db1 = dS1.sum(axis=0)  # H x 1

            # update
            W1 += -param_delta * dW1
            b1 += -param_delta * db1
            W2 += -param_delta * dW2
            b2 += -param_delta * db2

        # store parameters
        self.W1 = W1
        self.b1 = b1
        self.W2 = W2
        self.b2 = b2

    def fcann2_classify(self, X):
        S1 = np.dot(X, self.W1.T) + self.b1  # N x H
        H1 = np.maximum(0, S1)     # N X H
        S2 = np.dot(H1, self.W2.T) + self.b2   # N x C

        expscores = np.exp(S2)      # N x C
        sumexp = expscores.sum(axis=1) # N x 1

        probs = expscores / sumexp.reshape(-1,1)     # N x C
        return probs


