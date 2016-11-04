# coding: utf-8

from utils import save_submission,load_data
import numpy as np
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
from sklearn.metrics import classification_report, accuracy_score
import datetime
import sys
from softmaxSKL import softmaxModel
import pprint

# Output Results to File
now = datetime.datetime.now()
sys.stdout = open("Large/output" + str(now.hour) + str(now.minute) + ".txt", "w")

data_fn = "NOISY_MNIST_SUBSETS.h5"

# Load Model Data
Xlarge,Ylarge = load_data(data_fn, "large_train")
Xval,Yval = load_data(data_fn, "val")

# Load competition Data
kaggleX = load_data(data_fn, 'kaggle')

# Mean Center the Data
Xlarge = Xlarge - Xlarge.mean(0)
Xval = Xval - Xval.mean(0)
kaggleX = kaggleX - kaggleX.mean(0)

# -- training options; these are suggestions, feel free to experiment
bigOpt = {
    'eta': [0.6],   # initial learning rate
    'maxiter': [10000],  # max number of iterations (updates) of SGD
    'batch_size': [70, 60, 50], 
    'etadrop': [0.95], # when dropping eta, multiply it by this number (e.g., .5 means halve it)
    'eta_frac': [0.18, .2, 0.22],  # drop eta every eta_frac fraction of the max iterations
    'lambda_' : [0.015, 0.01, 0.05]                  # so if eta_frac is .2, and maxiter is 10000, drop eta every 2000 iterations
}

pprint.pprint(bigOpt, width=1)

gs = RandomizedSearchCV(softmaxModel(), bigOpt, cv=5, n_jobs=-1, 
    verbose=1, n_iter=144)
gs.fit(Xlarge, Ylarge)
print("Best parameters set found on development set:\n")
pprint.pprint(gs.best_params_, width=1)
# Test on validation
y_true, y_pred = Yval.argmax(-1), gs.predict(Xval)
print("\nAccuracy_Score")
print(accuracy_score(y_true, y_pred))
print(classification_report(y_true, y_pred))

print("\n Confusion Matrix\n")
print(confusion_matrix(y_true, y_pred))

# Test on custom softmax
# Kaggle 
kagglePrediction = gs.predict(kaggleX)
# Save results
save_submission('submission-large.csv',  kagglePrediction)


