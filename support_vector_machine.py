# numpy libraries
import numpy as np

# matplotlib libraries
import matplotlib.pyplot as plt

# scikit-learn libraries
from sklearn.dummy import DummyClassifier
from sklearn.svm import SVC
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn import metrics
from sklearn.utils import shuffle
import utils as util

from collections import defaultdict

filenameX = 'X.csv'
filenamey = 'y.csv'


def set_data_weights(y, data):
    sample_weights = []
    class_weights = 1.0 * len(y)/(2 * np.bincount(y.astype(int)))
    for sample in y.astype(int):
        sample_weights.append(class_weights[sample])
    data.weights = np.array(sample_weights)


def select_param_linear(X, y, kf, metric="accuracy") :
    """
    Sweeps different settings for the hyperparameter of a linear-kernel SVM,
    calculating the k-fold CV performance for each setting, then selecting the
    hyperparameter that 'maximize' the average k-fold CV performance.
    
    Parameters
    --------------------
        X      -- numpy array of shape (n,d), feature vectors
                    n = number of examples
                    d = number of features
        y      -- numpy array of shape (n,), binary labels {1,-1}
        kf     -- model_selection.KFold or model_selection.StratifiedKFold
        metric -- string, option used to select performance measure
        plot   -- boolean, make a plot
    
    Returns
    --------------------
        C -- float, optimal parameter value for linear-kernel SVM
    """
    
    print 'Linear SVM Hyperparameter Selection based on ' + str(metric) + ':'
    C_range = 10.0 ** np.arange(-3, 3)
    
    ### ========== TODO : START ========== ###
    # part 2c: select optimal hyperparameter using cross-validation
    scores = [0 for _ in xrange(len(C_range))] # dummy values, feel free to change
    
    for i, C in enumerate(C_range):
        clf = SVC(C=C, kernel="linear", class_weight="balanced")
        scores[i] = cv_performance(clf, X, y, kf, metric=metric)
    
    return C_range[np.argmax(scores)]

def select_param_rbf(X, y, kf, metric="accuracy") :
    """
    Sweeps different settings for the hyperparameters of an RBF-kernel SVM,
    calculating the k-fold CV performance for each setting, then selecting the
    hyperparameters that 'maximize' the average k-fold CV performance.
    
    Parameters
    --------------------
        X       -- numpy array of shape (n,d), feature vectors
                     n = number of examples
                     d = number of features
        y       -- numpy array of shape (n,), binary labels {1,-1}
        kf      -- model_selection.KFold or model_selection.StratifiedKFold
        metric  -- string, option used to select performance measure
    
    Returns
    --------------------
        C        -- float, optimal parameter value for an RBF-kernel SVM
        gamma    -- float, optimal parameter value for an RBF-kernel SVM
    """
    
    print 'RBF SVM Hyperparameter Selection based on ' + str(metric) + ':'
    
    ### ========== TODO : START ========== ###
    # part 3b: create grid, then select optimal hyperparameters using cross-validation
    C_range = 10.0 ** np.arange(0, 6)
    gamma_range = 10.0 ** np.arange(-5, 1)
    coef_range = np.arange(-2, 4)
    degree_range = np.arange(1, 6)
    scores = np.zeros((len(C_range), len(gamma_range), len(coef_range), len(degree_range))) # dummy values, feel free to change
    
    for i, C in enumerate(C_range):
        for j, gamma in enumerate(gamma_range):
            for k, coef in enumerate(coef_range):
                for l, degree in enumerate(degree_range):
                    clf = SVC(C=C, kernel="rbf", coef0=coef, degree=degree, gamma=gamma, class_weight="balanced")
                    scores[i][j][k][l] = cv_performance(clf, X, y, kf, metric=metric)

    Ci, gammaj, coefk, degreel = np.unravel_index(np.argmax(scores), scores.shape)

    print scores, Ci, gammaj, coefk, degreel
    
    return scores.max(), C_range[Ci], gamma_range[gammaj], coef_range[coefk], degree_range[degreel]



def cv_performance(clf, X, y, kf, metric="accuracy") :
    """
    Splits the data, X and y, into k-folds and runs k-fold cross-validation.
    Trains classifier on k-1 folds and tests on the remaining fold.
    Calculates the k-fold cross-validation performance metric for classifier
    by averaging the performance across folds.
    
    Parameters
    --------------------
        clf    -- classifier (instance of SVC)
        X      -- numpy array of shape (n,d), feature vectors
                    n = number of examples
                    d = number of features
        y      -- numpy array of shape (n,), binary labels {1,-1}
        kf     -- model_selection.KFold or model_selection.StratifiedKFold
        metric -- string, option used to select performance measure
    
    Returns
    --------------------
        score   -- float, average cross-validation performance across k folds
    """
    
    scores = []
    for train, test in kf.split(X, y) :
        X_train, X_test, y_train, y_test = X[train], X[test], y[train], y[test]
        clf.fit(X_train, y_train)
        # use SVC.decision_function to make ``continuous-valued'' predictions
        y_pred = clf.decision_function(X_test)
        score = performance(y_test, y_pred, metric)
        if not np.isnan(score) :
            scores.append(score)
    return np.array(scores).mean()


def performance(y_true, y_pred, metric="accuracy") :
    """
    Calculates the performance metric based on the agreement between the 
    true labels and the predicted labels.
    
    Parameters
    --------------------
        y_true -- numpy array of shape (n,), known labels
        y_pred -- numpy array of shape (n,), (continuous-valued) predictions
        metric -- string, option used to select the performance measure
                  options: 'accuracy', 'f1_score', 'auroc', 'precision',
                           'sensitivity', 'specificity'        
    
    Returns
    --------------------
        score  -- float, performance score
    """
    # map continuous-valued predictions to binary labels
    y_label = np.sign(y_pred)
    y_label[y_label==-1] = 0 # map points of hyperplane to +1
    
    ### ========== TODO : START ========== ###
    # part 2a: compute classifier performance
    score = 0.0
    matrix = metrics.confusion_matrix(y_true, y_label, labels=[1, -1])
    if metric == "accuracy":
        score = metrics.accuracy_score(y_true, y_label)
    elif metric == "f1_score":
        score = metrics.f1_score(y_true, y_label)
    elif metric == "auroc":
        score = metrics.roc_auc_score(y_true, y_pred)
    elif metric == "precision":
        score = metrics.precision_score(y_true, y_label)
    elif metric == "sensitivity":
        TP = matrix[0][0]
        FN = matrix[0][1]
        score = TP*1.0/(TP + FN)
    elif metric == "specificity":
        TN = matrix[1][1]
        FP = matrix[1][0]
        score = TN*1.0/(TN + FP)

    return score

def main():
    data = util.load_data(filenameX, filenamey, header=1)

    X, y = data.X, data.y
    print data.Xnames
    set_data_weights(y, data)

    n_splits = 5
    kf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=None)

    metric_list = ["accuracy", "f1_score", "auroc", "precision", "sensitivity", "specificity"]

    max_f1_linear = 0

    # for train, test in kf.split(X, y):
    #     X_train, X_test, y_train, y_test = X[train], X[test], y[train], y[test]
    #     train_weights, test_weights = data.weights[train], data.weights[test]

    #     print select_param_rbf(X_train, y_train, kf, metric="f1_score")

    #     max_f1_linear = max(max_f1_linear, select_param_linear(X, y, kf, metric="f1_score"))

    print 'max_f1_linear', max_f1_linear

    C, gamma = (10.0, 0.1)

    X_train, X_test, y_train, y_test, weight_train, weight_test = train_test_split(X, y, data.weights, test_size=0.2, stratify=y)


    dumclf = DummyClassifier(strategy="most_frequent")
    dumclf.fit(X_train, y_train, sample_weight=weight_train)

    rbfclf = SVC(C=C, gamma=gamma, class_weight="balanced")
    rbfclf.fit(X_train, y_train)
    y_pred = rbfclf.predict(X_test)

    # compute classifier performance
    for metric in metric_list:
        print metric + ":", performance(y_test, y_pred, metric)

    svc_train_score = rbfclf.score(X_train, y_train)
    svc_test_score = rbfclf.score(X_test, y_test)
    dummy_train_score = dumclf.score(X_train, y_train, weight_train)
    dummy_test_score = dumclf.score(X_test, y_test, weight_test)

    print metrics.confusion_matrix(y_test, y_pred, labels=[1, 0], sample_weight=weight_test)

    print "RBFSVC train accuracy: %.6f" % (svc_train_score)
    print "Dummy train accuracy: %.6f" % (dummy_train_score)
    print "RBFSVC test accuracy: %.6f" % (svc_test_score)
    print "Dummy test accuracy: %.6f" % (dummy_test_score)


    max_f1_linear = 10.0
    linclf = SVC(C=max_f1_linear, kernel="linear")

    linclf.fit(X_train, y_train)

    print "Linear SVC train accuracy: %.6f" % (linclf.score(X_train,y_train))
    print "Linear SVC test accuracy: %.6f" % (linclf.score(X_test,y_test))

    y_pred = linclf.predict(X_test)

    print "Linear SVC test F1 score: %.6f" % (performance(y_test, y_pred, metric="f1_score"))

    print "Top ten features (probably) in order from largest to smallest:"
    indices = linclf.coef_[0].argsort()[-10:][::-1]
    print [data.Xnames[i] for i in indices]


    print "RBF F1/accuracy score with each of the top ten features removed."
    for i in indices:
        X_train_mod = np.delete(X_train, i, 1)
        X_test_mod = np.delete(X_test, i, 1)
        rbfclf.fit(X_train_mod, y_train)
        y_train_pred = rbfclf.predict(X_train_mod)
        y_test_pred = rbfclf.predict(X_test_mod)

        print "%s:" % (data.Xnames[i])
        print "\tAccuracy: "
        print "\t\tTrain: %.6f" % (rbfclf.score(X_train_mod, y_train))
        print "\t\tTest: %.6f" % (rbfclf.score(X_test_mod, y_test))
        print "\tF1 Score:"
        print "\t\tTrain: %.6f" % (performance(y_train, y_train_pred, metric="f1_score"))
        print "\t\tTest: %.6f" % (performance(y_test, y_test_pred, metric="f1_score"))


    print "RBF Least predictive ten features (probably) in order from smallest to largest:"
    indices = linclf.coef_[0].argsort()[:10]
    print [data.Xnames[i] for i in indices]

    print "F1/accuracy score with each of the bottom ten features removed cumulatively"
    for i in indices:
        X_train_mod = np.delete(X_train, indices[:i+1], 1)
        X_test_mod = np.delete(X_test, indices[:i+1], 1)
        rbfclf.fit(X_train_mod, y_train)
        y_train_pred = rbfclf.predict(X_train_mod)
        y_test_pred = rbfclf.predict(X_test_mod)

        print "%s:" % (data.Xnames[i])
        print "\tAccuracy: "
        print "\t\tTrain: %.6f" % (rbfclf.score(X_train_mod, y_train))
        print "\t\tTest: %.6f" % (rbfclf.score(X_test_mod, y_test))
        print "\tF1 Score:"
        print "\t\tTrain: %.6f" % (performance(y_train, y_train_pred, metric="f1_score"))
        print "\t\tTest: %.6f" % (performance(y_test, y_test_pred, metric="f1_score"))

if __name__ == "__main__" :
    main()