# coding: utf-8

import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin

class softmaxModel(BaseEstimator, ClassifierMixin):  
    """An example of classifier"""


    def __init__(self, eta=2, maxiter=10000, batch_size=20, etadrop=0.5, eta_frac=0.2, lambda_=0.001):

        self.eta = eta
        self.maxiter = maxiter
        self.batch_size = batch_size
        self.etadrop = etadrop
        self.eta_frac = eta_frac
        self.lambda_ = lambda_

        CLASSES = 10
        NFEATURES = 576 

        seed = np.random.RandomState(2341)  # to make sure everyone starts from the same point
        random_init = seed.normal(scale=0.01, size=(NFEATURES, CLASSES)) 
        self.model = { 'weight': random_init, 'bias': np.zeros(CLASSES)}


    def _classProbs(self, X):
        
        model = self.model
        Z = np.dot(X, model['weight']) + model['bias']
        # Softmax
        val = np.exp(Z)
        rv = val/val.sum(axis=1, keepdims=True)
        return rv

    def predict(self, X):
        """
        Compute hard label assignments based on model predictions, and return the accuracy vector
        Input:
        X : N x d array of data (no constant term)
        Output:
        yHat : N x 1
        """
        yHat = self._classProbs(X).argmax(-1)
        return yHat

    def score(self, X, Y):
        """
        Compute error rate (between 0 and 1) for the model
        """
        score = self.predict(X) == Y.argmax(-1)
        return score.mean()

    def fit(self, X, Y):
        """
        Run the train + evaluation on a given train/val partition
        trainopt: various (hyper)parameters of the training procedure
        """
        
        N = X.shape[0] # number of data points in X
        
        shuffled_idx = np.random.permutation(N)
        start_idx = 0
        for iteration in range(self.maxiter):
            if iteration % int(self.eta_frac * self.maxiter ) == 0:
                self.eta *= self.etadrop 
            # Form the next mini-batch
            stop_idx = min(start_idx + self.batch_size , N)
            batch_idx = range(N)[int(start_idx):int(stop_idx)]
            bX = X[shuffled_idx[batch_idx],:]
            bY = Y[shuffled_idx[batch_idx],:]
            # Update Model
            self._modelUpdate(bX, bY)
            # Update batch index
            start_idx = stop_idx % N
        
        return self

    def _modelUpdate(self, X, Y):
        """
        Update the model
        Input:
        X, Y : the inputs and 1-hot encoded labels
        """
        bias = self.model["bias"]
        weight = self.model["weight"]
        eta = self.eta
        num_exp = X.shape[0]
        probs = self._classProbs(X)
        
        scores = probs - Y
        scores /= num_exp

        grad_W = np.dot(X.T, scores)
        grad_b = np.sum(scores, axis=0, keepdims=True) 

        weight += -eta * (grad_W + (self.lambda_ * weight))
        bias += -eta * grad_b[0]

