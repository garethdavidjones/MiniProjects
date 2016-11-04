# coding: utf-8

from utils import save_submission,load_data
# SmallSoftmax.py

import numpy as np
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
import datetime
import sys
from softmaxSKL import softmaxModel
import pprint

now = datetime.datetime.now()
sys.stdout = open("Small/output" + str(now.hour) + str(now.minute) + ".txt", "w")

data_fn = "NOISY_MNIST_SUBSETS.h5"

# Load Model Data
Xsmall,Ysmall = load_data(data_fn, "small_train")
Xval,Yval = load_data(data_fn, "val")

# Load competition Data
kaggleX = load_data(data_fn, 'kaggle')

# Mean Center the Data
Xsmall = Xsmall - Xsmall.mean(0)
Xval = Xval - Xval.mean(0)
kaggleX = kaggleX - kaggleX.mean(0)

smallOpt = {
    'eta': [2, 1.5, 1],   # initial learning rate
    'maxiter': [10000],  # max number of iterations (updates) of SGD
    'batch_size': [2, 3, 4], 
    'etadrop': [0.4, 0.3, 0.2], # when dropping eta, multiply it by this number (e.g., .5 means halve it)
    'eta_frac': [0.8, 0.7],  # drop eta every eta_frac fraction of the max iterations
    'lambda_' : [0.1, 0.05, .025]                   # so if eta_frac is .2, and maxiter is 10000, drop eta every 2000 iterations
}

pprint.pprint(smallOpt, width=1)

gs = GridSearchCV(softmaxModel(), smallOpt, cv=5, n_jobs=-1, 
                        verbose=1)
gs.fit(Xsmall, Ysmall)
print("Best parameters set found on development set:\n")
pprint.pprint(gs.best_params_, width=1)

# Test on validation
y_true, y_pred = Yval.argmax(-1), gs.predict(Xval)
print("\nAccuracy_Score")
print(accuracy_score(y_true, y_pred))
print(classification_report(y_true, y_pred))

print("\n Confusion Matrix")
print(confusion_matrix(y_true, y_pred))
# Test on custom softmax
# Kaggle 
kagglePrediction = gs.predict(kaggleX)
# Save results
save_submission('submission-small.csv',  kagglePrediction)



