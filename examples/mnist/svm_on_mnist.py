import numpy as np
from sklearn.svm import SVC
from sklearn.datasets import load_svmlight_file
import os

def evaluation(size, log_C, gamma):
    path = os.path.join(os.path.dirname(os.path.abspath(__file__)), os.pardir)
    path = os.path.abspath(os.path.join(path, "mnist_data"))
    print "loading train set..."
    X, y = load_svmlight_file(os.path.join(path, "mnist"))
    idx = np.arange(0, X.shape[0])
    np.random.shuffle(idx)
    idx = idx[:size]
    X_train = X[idx, :]
    y_train = y[idx]
    C_param = 2 ** log_C
    print "number of chosen data points: " + str(size)

    clf = SVC(C=C_param,  gamma=gamma)
    
    print "training..."
    clf.fit(X_train, y_train)
    print "evaluating (on training set!)..."
    score = clf.score(X, y)

    print "Mean Accuracy: " + str(score)

    return 1 - score


def main(job_id, params):
    print 'Parameters: '
    print params
    return evaluation(params['Size'], params['C'], params['gamma'])