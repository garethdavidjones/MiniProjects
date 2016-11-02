# coding: utf-8

from utils import save_submission,load_data
import numpy as np
import copy
from sklearn.model_selection import GridSearchCV
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.metrics import classification_report 
import datetime
now = datetime.datetime.now()
import sys
sys.stdout = open("Small/output"+str(now.hour) + str(now.minute) +".txt", "w")

data_fn = "NOISY_MNIST_SUBSETS.h5"

Xsmall,Ysmall = load_data(data_fn, "small_train")
Xsmall = Xsmall - Xsmall.mean(0)

Xlarge,Ylarge = load_data(data_fn, "large_train")
Xlarge = Xlarge - Xlarge.mean(0)

Xval,Yval = load_data(data_fn, "val")
Xval = Xval - Xval.mean(0)

kaggleX = load_data(data_fn, 'kaggle')

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
        
    def predict(self, X):
        """
        Evaluate the soft predictions of the model.
        Input:
        X : N x d array (no unit terms)
        model : dictionary containing 'weight' and 'bias'
        Output:
        yhat : N x C array
            yhat[n][:] contains the softmax posterior distribution over C classes for X[n][:]
        """
        
        model = self.model
        Z = np.dot(X, model['weight']) + model['bias']
        # Softmax
        val = np.exp(Z)
        val/val.sum(axis=1, keepdims=True)
        return val


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
        Y : N x C array with 1-hot encoding of true labels
        model: dictionary 
        Output:
        acc : N array of errors, acc[n] is 1 if correct and 0 otherwise
        """

        return self._classProbs(X).argmax(-1)

    def score(self, X, Y):
        """
        Compute error rate (between 0 and 1) for the model
        """
        scores = self._classProbs(X).argmax(-1) == Y.argmax(-1)
        return scores.mean()


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
        model : the currrent model
        lambda : regularization coefficient for L2 penalty
        eta : learning rate
        Output:
        updated model
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



smallOpt = {
    'eta': [2.5, 2.0, 1.5, 1.0],   # initial learning rate
    'maxiter': [10000, 8000],  # max number of iterations (updates) of SGD
    'batch_size': [1, 3, 8], 
    'etadrop': [0.65, 0.5, 0.3], # when dropping eta, multiply it by this number (e.g., .5 means halve it)
    'eta_frac': [ 0.6, 0.5, 0.4, 0.3],  # drop eta every eta_frac fraction of the max iterations
    'lambda_' :[0.01, 0.0075, 0.0125, 0.005]                   # so if eta_frac is .2, and maxiter is 10000, drop eta every 2000 iterations
}




#smallOpt = {
 #   'eta': [2.5],   # initial learning rate
 #   'maxiter': [15000],  # max number of iter
#    'batch_size': [1], 
#    'etadrop': [0.65], # when dropping e
 #   'eta_frac': [0.8],  # drop
  #  'lambda_' : [0.01]                   #

#}




print(smallOpt)

gs = GridSearchCV(softmaxModel(), smallOpt, cv=5, n_jobs=10)
gs.fit(Xsmall, Ysmall)
print("Grid scores on development set:")
print("Best parameters set found on development set:\n")
print(gs.best_params_, "\n")
y_true, y_pred = Yval.argmax(-1), gs.predict(Xval)
print(classification_report(y_true, y_pred) )

    # Save results
save_submission('submission-small.csv',  gs.predict(kaggleX))


