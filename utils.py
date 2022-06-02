import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import math
import os
from preprocessing import *


class Data :

    def __init__(self, X=None, y=None, Xnames=None, yname=None, weights=None) :
        """
        Data class.

        Attributes
        --------------------
            X       -- numpy array of shape (n,d), features
            y       -- numpy array of shape (n,), targets
        """

        # n = number of examples, d = dimensionality
        self.X = X
        self.y = y
        self.Xnames = Xnames
        self.yname = yname
        self.weights = weights

    def load(self, filenameX, filenamey, header=0) :
        """
        Load csv file into X array of features and y array of labels.

        Parameters
        --------------------
            filename   -- string, filename
        """

        # determine filename
        dir = os.path.dirname(__file__)
        fX = os.path.join(dir, 'Dataset', filenameX)
        fy = os.path.join(dir, 'Dataset', filenamey)

        # load data
        with open(fX, 'r') as fid :
            dataX = np.loadtxt(fid, delimiter=",", skiprows=header)

        with open(fy, 'r') as fid :
            datay = np.loadtxt(fid, delimiter=",", skiprows=header)


        # separate features and labels
        self.X = dataX
        self.y = datay

        # load feature and label names
        if header != 0:
            with open(fX, 'r') as fid :
                headerX = fid.readline().rstrip().split(",")

            with open(fy, 'r') as fid :
                headery = fid.readline().rstrip().split(",")

            self.Xnames = headerX
            self.yname = headery

        else:
            self.Xnames = None
            self.yname = None

    def plot(self) :
        """Plot features and labels."""
        pos = np.nonzero(self.y > 0)  # matlab: find(y > 0)
        neg = np.nonzero(self.y < 0)  # matlab: find(y < 0)
        plt.plot(self.X[pos,0], self.X[pos,1], 'b+', markersize=5)
        plt.plot(self.X[neg,0], self.X[neg,1], 'ro', markersize=5)
        plt.show()

    def plot_histogram(self) :
        """
        Plots histogram of values in X grouped by y.

        Parameters
        --------------------
            X     -- numpy array of shape (n,), feature values
            y     -- numpy array of shape (n,), target classes
            Xnames -- string, name of feature
            yname -- string, name of target
        """
        for i in xrange(len(self.Xnames)):
            # set up data for plotting
            targets = sorted(set(self.y))
            data = []; labels = []
            for target in targets :
                features = [self.X[:,i][j] for j in xrange(len(self.y)) if self.y[j] == target]
                data.append(features)
                labels.append('%s = %s' % (self.yname[0], target))

            # set up histogram bins
            features = set(self.X[:,i])
            nfeatures = len(features)
            test_range = range(int(math.floor(min(features))), int(math.ceil(max(features)))+1)
            if nfeatures < 33 and sorted(features) == test_range:
                bins = test_range + [test_range[-1] + 1] # add last bin
                align = 'left'
            else :
                bins = 33
                align = 'mid'

            # plot
            plt.figure()
            n, bins, patches = plt.hist(data, bins=bins, align=align, alpha=0.5, label=labels, log=True, density=False)
            # n, bins, patches = plt.hist(data, density=True)
            plt.xlabel(self.Xnames[i])
            plt.ylabel('Frequency')
            plt.legend()
            plt.legend(loc='upper left')
            plt.show()

def save_data(filename, df):
    df.to_csv(filename)

# helper functions
def load_data(filenameX, filenamey, header=0) :
    """Load csv file into Data class."""
    data = Data()
    data.load(filenameX, filenamey, header=header)
    return data

def update_data():
    npX, npy, Xnames, X, y = main()
    X.to_csv('Dataset/X.csv', index=False)
    y.to_csv('Dataset/y.csv', index=False, header=True)
