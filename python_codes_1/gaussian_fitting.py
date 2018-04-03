from osgeo import gdal
import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import math
import incidence_angle_corr
from math import pi
import itertools
from scipy import linalg
from sklearn import mixture
#step1: selection of features
#step2: feature transformation using histogram
#step3: 

color_iter = itertools.cycle(['navy', 'c', 'cornflowerblue', 'gold', 'darkorange'])

#for i in color_iter:
    #print(i)

def samajh():
    np.random.seed(0)
    C = np.array([[0., -0.1], [1.7, .4]])
    n_samples = 500
    X_0=np.random.randn(n_samples, 2)
    X_1=np.dot(np.random.randn(n_samples, 2),C)
    X_2=np.random.randn(n_samples, 2) + np.array([-6, 3])
    X = np.r_[np.dot(np.random.randn(n_samples, 2), C), .7 * np.random.randn(n_samples, 2) + np.array([-6, 3])]
    print(X.shape)
    #plt.hist(X,bins=150, rwidth=0.5, density=False, histtype='stepfilled')
    plt.scatter(X[:, 0], X[:, 1], .8)
    plt.show()


def plot_results(X, Y_, means, covariances, index, title):
    splot = plt.subplot(2, 1, 1 + index)
    for i, (mean, covar, color) in enumerate(zip(
            means, covariances, color_iter)):
        v, w = linalg.eigh(covar)
        v = 2. * np.sqrt(2.) * np.sqrt(v)
        u = w[0] / linalg.norm(w[0])
        # as the DP will not use every component it has access to
        # unless it needs it, we shouldn't plot the redundant
        # components.
        if not np.any(Y_ == i):
            continue
        plt.scatter(X[Y_ == i, 0], X[Y_ == i, 1], .8, color=color)

        # Plot an ellipse to show the Gaussian component
        angle = np.arctan(u[1] / u[0])
        angle = 180. * angle / np.pi  # convert to degrees
        ell = mpl.patches.Ellipse(mean, v[0], v[1], 180. + angle, color=color)
        ell.set_clip_box(splot.bbox)
        ell.set_alpha(0.2)
        splot.add_artist(ell)

    plt.xlim(-9., 5.)
    plt.ylim(-3., 6.)
    plt.xticks(())
    plt.yticks(())
    plt.title(title)

def from_gmm_site():
    # Number of samples per component
    n_samples = 500

    # Generate random sample, two components
    np.random.seed(0)
    C = np.array([[0., -0.1], [1.7, .4]])
    X = np.r_[np.dot(np.random.randn(n_samples, 2), C),
            .7 * np.random.randn(n_samples, 2) + np.array([-6, 3])]
    print(X.shape)
    # Fit a Gaussian mixture with EM using five components
    gmm = mixture.GaussianMixture(n_components=1, covariance_type='full').fit(X)
    print(gmm.means_)
    plot_results(X, gmm.predict(X), gmm.means_, gmm.covariances_, 0,
                'Gaussian Mixture')

    # Fit a Dirichlet process Gaussian mixture using five components
    dpgmm = mixture.BayesianGaussianMixture(n_components=5,
                                            covariance_type='full').fit(X)
    plot_results(X, dpgmm.predict(X), dpgmm.means_, dpgmm.covariances_, 1,
                'Bayesian Gaussian Mixture with a Dirichlet process prior')

    plt.show()
def count(start=0, step=1):
    # count(10) --> 10 11 12 13 14 ...
    # count(2.5, 0.5) -> 2.5 3.0 3.5 ...
    n = start
    while True:
        yield n
        n += step

if __name__=='__main__':
    #a=count()
    #print(next(a))
    #print(next(a))
    #print(next(a))
    samajh()
    
    