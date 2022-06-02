"""
Adapted from digits.py in hw7
"""

# python libraries
import collections

# numpy libraries
import numpy as np

# matplotlib libraries
import matplotlib.pyplot as plt

# scikit-learn libraries
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import BaggingClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import train_test_split
from sklearn import metrics

# file with helper functions
import utils


######################################################################
# Set Feature Weights
######################################################################

def set_data_weights(y, data):
    sample_weights = []
    class_weights = 1.0 * len(y)/(2 * np.bincount(y.astype(int)))
    for sample in y.astype(int):
        sample_weights.append(class_weights[sample])
    data.weights = np.array(sample_weights)


######################################################################
# bagging functions
######################################################################

def bagging_ensemble(X_train, y_train, X_test, y_test, max_features=None, num_clf=11) :
    """
    Compute performance of bagging ensemble classifier.

    Parameters
    --------------------
        X_train      -- numpy array of shape (n_train,d), training features
        y_train      -- numpy array of shape (n_train,),  training targets
        X_test       -- numpy array of shape (n_test,d),  test features
        y_test       -- numpy array of shape (n_test,),   test targets
        max_features -- int, number of features to consider when looking for best split
        num_clf      -- int, number of decision tree classifiers in bagging ensemble

    Returns
    --------------------
        accuracy     -- float, accuracy of bagging ensemble classifier on test data
    """
    base_clf = DecisionTreeClassifier(criterion='entropy', max_features=max_features)
    clf = BaggingClassifier(base_clf, n_estimators=num_clf)
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    return clf, metrics.f1_score(y_test, y_pred)


def random_forest(X_train, y_train, X_test, y_test, max_features, num_clf=11,
                  bagging=bagging_ensemble) :
    """
    Wrapper around bagging_ensemble to use feature-limited decision trees.

    Additional Parameters
    --------------------
        bagging      -- bagging_ensemble
    """
    return bagging(X_train, y_train, X_test, y_test,
                    max_features=max_features, num_clf=num_clf)


######################################################################
# plotting functions
######################################################################

def plot_scores(max_features, bagging_scores, random_forest_scores) :
    """
    Plot values in random_forest_scores and bagging_scores.
    (The scores should use the same set of 100 different train and test set splits.)

    Parameters
    --------------------
        max_features         -- list, number of features considered when looking for best split
        bagging_scores       -- list, accuracies for bagging ensemble classifier using DTs
        random_forest_scores -- list, accuracies for random forest classifier
    """

    plt.figure()
    plt.plot(max_features, bagging_scores, '--', label='bagging')
    plt.plot(max_features, random_forest_scores, '--', label='random forest')
    plt.xlabel('max features considered per split')
    plt.ylabel('F1 Score')
    plt.legend(loc='upper right')
    plt.show()


def plot_histograms(bagging_scores, random_forest_scores):
    """
    Plot histograms of values in random_forest_scores and bagging_scores.
    (The scores should use the same set of 100 different train and test set splits.)

    Parameters
    --------------------
        bagging_scores       -- list, accuracies for bagging ensemble classifier using DTs
        random_forest_scores -- list, accuracies for random forest classifier
    """

    bins = np.linspace(0.8, 1.0, 100)
    plt.figure()
    plt.hist(bagging_scores, bins, alpha=0.5, label='bagging')
    plt.hist(random_forest_scores, bins, alpha=0.5, label='random forest')
    plt.xlabel('accuracy')
    plt.ylabel('frequency')
    plt.legend(loc='upper left')
    plt.show()

##################################################################
# functions to test performance and find optimal hyperparameters
##################################################################

def findHyperParam(filenameX, filenamey):
    # below is code from hw7 that may be useful in the future
    # so it's commented out for now

    # load dataset
    data = utils.load_data(filenameX, filenamey, header=1)
    X = data.X
    y = data.y

    # evaluation parameters
    num_trials = 100

    # sklearn or home-grown bagging ensemble
    bagging = bagging_ensemble

    #========================================
    # vary number of features

    # calculate accuracy of bagging ensemble and random forest
    #   for 100 random training and test set splits
    # make sure to use same splits to enable proper comparison
    max_features_vector = range(1,34, 2)
    bagging_scores = []
    random_forest_scores = collections.defaultdict(list)
    for i in range(num_trials):
        print i
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
        bagging_scores.append(bagging(X_train, y_train, X_test, y_test)[1])
        for m in max_features_vector :
            random_forest_scores[m].append(random_forest(X_train, y_train, X_test, y_test, max_features = m,
                                                         bagging=bagging)[1])

    # analyze how performance of bagging and random forest changes with m
    bagging_results = []
    random_forest_results = []
    for m in max_features_vector :
        bagging_results.append(np.median(np.array(bagging_scores)))
        random_forest_results.append(np.median(np.array(random_forest_scores[m])))
    plot_scores(max_features_vector, bagging_results, random_forest_results)

    return random_forest_results[np.argmax(random_forest_results)]


def performance(y_true, y_pred, metric="f1_score") :
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

    # compute classifier performance
    score = 0.0
    matrix = metrics.confusion_matrix(y_true, y_label, labels=[1, 0])
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


def cv_performance(clf, X, y, kf, metric="f1_score") :
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
        y_pred = clf.predict(X_test)
        score = performance(y_test, y_pred, metric)
        if not np.isnan(score) :
            scores.append(score)
    return np.array(scores).mean()


def select_params_rf(X, y, kf, metric="f1_score"):
    """
    Sweeps different settings for the hyperparameters of a RF,
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
        opt_score         -- best score from the different hyperparameter combinatiom
        n_estimators      -- int, optimal number of trees in the forest
        max_features      -- int, optimal number of features to consider for each split
    """
    
    print 'RF Hyperparameter Selection based on ' + str(metric) + ':'
    
    # select num_estimators and max_features for now (make two arrays or lists/ranges)

    min_num_trees = 50
    max_num_trees = 100

    min_max_features = 10
    max_max_features = X.shape[1]

    opt_score = 0.0
    opt_n_estimators = 0
    opt_max_features = 0

    for i in range(min_num_trees, max_num_trees+1):
        for j in range(min_max_features, max_max_features+1):
            clf = RandomForestClassifier(n_estimators=i, criterion="entropy", 
                                         max_features=j, class_weight="balanced")
            score = cv_performance(clf, X, y, kf, metric)
            if score >= opt_score:
                opt_score = score
                opt_n_estimators = i
                opt_max_features = j
    
    return opt_score, opt_n_estimators, opt_max_features


def select_params_rf_using_oob(X, y):
    """
    Sweeps different settings for the hyperparameters of a RF,
    calculating the oob_score, then selecting the
    hyperparameters that 'maximize' this oob_score.
    
    Parameters
    --------------------
        X       -- numpy array of shape (n,d), feature vectors
                     n = number of examples
                     d = number of features
        y       -- numpy array of shape (n,), binary labels {1,-1}
    
    Returns
    --------------------
        n_estimators      -- int, optimal number of trees in the forest
        max_features      -- int, optimal number of features to consider for each split
        min_samples_leaf  -- int, optimal number of minimum samples required to form a leaf
    """

    # more trees improves performance, but also slows down code
    # want a large number but not too large so that the machine can handle it
    min_num_trees = 40
    max_num_trees = 75

    # more features considered improves performance generally
    # more options might decrease diversity of trees however, and thus performance
    # speed decreases as max_features increases
    min_max_features = 10
    max_max_features = X.shape[1]

    # smaller leafs means model more prone to noisy train data
    # want to find an optimum minimum leaf size
    min_samples_leaf_options = [1,25,50,75,100]

    max_oob_score = 0.0
    max_i = 0
    max_j = 0
    max_k = 0
    
    for i in range(min_num_trees, max_num_trees+1):
        for j in range(min_max_features, max_max_features+1):
            for k in min_samples_leaf_options:
                rf = RandomForestClassifier(n_estimators=i, criterion="entropy", max_features=j,
                                            min_samples_leaf=k, oob_score=True, class_weight="balanced")
                rf.fit(X,y)
                if rf.oob_score_ > max_oob_score:
                    max_i = i
                    max_j = j
                    max_k = k
                    max_oob_score = rf.oob_score_

    return max_i, max_j, max_k


def main():
    np.random.seed(1234)

    filenameX = 'X.csv'
    filenamey = 'y.csv'

    data = utils.load_data(filenameX, filenamey, header=1)

    X, y = data.X, data.y
    # uncomment the following line to see the names of all the features
    # print data.Xnames
    set_data_weights(y, data)

    n,d = X.shape

    """
    params considered for rf:
        n_estimators (find optimal)
        criterion ("entropy")
        max_features (find optimal)
        min_samples_leaf (find optimal)
        oob_score (True)
        class_weight ("balanced")
    """

    metric_list = ["accuracy", "f1_score", "auroc", "precision", "sensitivity", "specificity"]
    
    X_train, X_test, y_train, y_test, weight_train, weight_test = train_test_split(X, y, data.weights, test_size=0.2, stratify=y)

    """
    # this section is for finding the optimal hyperparameters using kfold cross validation
    # uncomment this block if you want to find hyperparameters this way and probably take a long time

    n_splits = 5
    kf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=None)

    score, n_estimators, max_features = select_params_rf(X, y, kf, metric="f1_score")

    print score
    print n_estimators
    print max_features
    """
    
    """
    # this section is for finding the optimal hyperparameters using rf oob_score
    # uncomment this block if you want to find hyperparameters this way

    # find optimal hyperparameters using oob_score
    n_estimators, max_features, min_samples_leaf = select_params_rf_using_oob(X,y)

    print n_estimators
    print max_features
    print min_samples_leaf
    """

    # the following section is for finding the performance and most/least important features
    # after seclecting the optimal hyperparameters using one of the methods above
    #
    # the optimal hyperparameters below were found using select_params_rf_using_oob
    n_estimators = 49
    max_features = 10
    min_samples_leaf = 1
    rf = RandomForestClassifier(n_estimators=n_estimators, criterion="entropy", max_features=max_features,
    	                        min_samples_leaf=min_samples_leaf, oob_score=True, class_weight="balanced")

    rf.fit(X_train, y_train)
    y_pred = rf.predict(X_test)
    y_pred_train = rf.predict(X_train)

    for metric in metric_list:
        print metric + ":", performance(y_test, y_pred, metric)

    rf_train_score = rf.score(X_train, y_train)
    rf_test_score = rf.score(X_test, y_test)
    rf_test_f1_score = performance(y_test, y_pred, metric="f1_score")
    rf_train_f1_score = performance(y_train, y_pred_train, metric="f1_score")

    print "RF train accuracy: %.6f" % (rf_train_score)
    print "RF test accuracy: %.6f" % (rf_test_score)
    print "RF train F1 score: %.6f" % (rf_train_f1_score)
    print "RF test F1 score: %.6f" % (rf_test_f1_score)

    
    # for confusion matrix
    y_label = np.sign(y_pred)
    y_label[y_label==-1] = 0 # map points of hyperplane to +1

    matrix = metrics.confusion_matrix(y_test, y_label, labels=[1, 0])
    print matrix
    

    
    # baseline classifier
    base_clf = DecisionTreeClassifier()
    clf = BaggingClassifier(base_clf)
    clf.fit(X_train, y_train)
    y_pred_base = clf.predict(X_test)
    y_pred_train_base = clf.predict(X_train)

    clf_train_score = clf.score(X_train, y_train)
    clf_test_score = clf.score(X_test, y_test)
    clf_test_f1_score = performance(y_test, y_pred_base, metric="f1_score")
    clf_train_f1_score = performance(y_train, y_pred_train_base, metric="f1_score")

    print "Baseline train accuracy: %.6f" % (clf_train_score)
    print "Baseline test accuracy: %.6f" % (clf_test_score)
    print "Baseline train F1 score: %.6f" % (clf_train_f1_score) 
    print "Baseline test F1 score: %.6f" % (clf_test_f1_score) 
    

    
    # feature importance
    feature_indices = [i for i in range(d)]
    feature_importances = rf.feature_importances_
    index_and_importance = zip(feature_indices, feature_importances)
    small_to_large = sorted(index_and_importance, key=lambda tup: tup[1])
    large_to_small = small_to_large[::-1]
    all_features_in_order = [(data.Xnames[tup[0]], tup[1]) for tup in large_to_small]


    print "\n"
    print "RF feature importance in order from largest to smallest:"
    print all_features_in_order
    print "\n"

    large_to_small_indices = [tup[0] for tup in large_to_small[:10]]
    small_to_large_indices = [tup[0] for tup in small_to_large[:10]]

    print "RF top ten features (probably) in order from largest to smallest:"
    print [data.Xnames[i] for i in large_to_small_indices]

    print "RF F1/accuracy score with each of the top ten features removed."
    for i in large_to_small_indices:
        X_train_mod = np.delete(X_train, i, 1)
        X_test_mod = np.delete(X_test, i, 1)
        rf.fit(X_train_mod, y_train)
        y_train_pred = rf.predict(X_train_mod)
        y_test_pred = rf.predict(X_test_mod)

        print "%s:" % (data.Xnames[i])
        print "\tAccuracy: "
        print "\t\tTrain: %.6f" % (rf.score(X_train_mod, y_train))
        print "\t\tTest: %.6f" % (rf.score(X_test_mod, y_test))

        print "\tF1 Score:"
        print "\t\tTrain: %.6f" % (performance(y_train, y_train_pred, metric="f1_score"))
        print "\t\tTest: %.6f" % (performance(y_test, y_test_pred, metric="f1_score"))


    print "\n"
    print "RF least predictive ten features (probably) in order from smallest to largest:"
    print [data.Xnames[i] for i in small_to_large_indices]

    print "RF F1/accuracy score with each of the bottom ten features removed cumulatively"
    for i in small_to_large_indices:
        X_train_mod = np.delete(X_train, small_to_large_indices[:i+1], 1)
        X_test_mod = np.delete(X_test, small_to_large_indices[:i+1], 1)
        rf.fit(X_train_mod, y_train)
        y_train_pred = rf.predict(X_train_mod)
        y_test_pred = rf.predict(X_test_mod)

        print "%s:" % (data.Xnames[i])
        print "\tAccuracy: "
        print "\t\tTrain: %.6f" % (rf.score(X_train_mod, y_train))
        print "\t\tTest: %.6f" % (rf.score(X_test_mod, y_test))
        print "\tF1 Score:"
        print "\t\tTrain: %.6f" % (performance(y_train, y_train_pred, metric="f1_score"))
        print "\t\tTest: %.6f" % (performance(y_test, y_test_pred, metric="f1_score"))
    


if __name__ == "__main__" :
    main()