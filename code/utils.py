import numpy as np
from numpy.random import normal, uniform
from numpy.linalg import inv

import matplotlib.pyplot as plt
from scipy.stats import norm, multivariate_normal

def polynomial(x, n):
    """
    Returns a numpy.array of powers of x of length n.
    x: polynomial variable
    n: degree of the polynomial
    """
    
    if hasattr(x, "__len__"):
        return np.array([[e**(i) for i in range(n+1)] for e in x])
    return np.array([x**(i) for i in range(n+1)]).reshape(1, -1)
    

def generateData(w, n, noise_stddev=0):
    """
    Returns a dataset of observations [x, t] uniformly sampled from the curve t = X.w + noise.
    w: parameters of the curve from which to take the observations
    n: number of observations to take
    noise_stddev: gaussian noise added to the targets
    """
    
    x = uniform(0.95, 5.05, n)
    x.sort()
    X = polynomial(x, len(list(w))-1)
    if noise_stddev != 0:
        t = X.dot(w) + normal(0, noise_stddev, n)
    else:
        t = X.dot(w)
    return np.array([x, t])

def getTarget(w, x):
    """
    Returns the target corresponding to x by performing t = X.w.
    w: parameters of the curve we are predicting
    x: attributes to get corresponding targets
    """

    X = polynomial(x, len(list(w))-1)
    return X.dot(w)

def getOptimalParams(x, t, n):
    """
    Returns the optimal parameters using the least squares fit.
    x, t: data to fit
    n: degree of the polynomial to fit
    """
    
    X = polynomial(x, n)
    XT = X.transpose()
    
    return inv(XT.dot(X)).dot(XT).dot(t)

def MSEloss(x, t, w):
    """
    Returns the Mean Squared Error calculated on the parameters.
    """
    
    X = polynomial(x, len(list(w))-1)
    tmp = t - X.dot(w)
    
    if len(tmp) > 1:
        return tmp.dot(tmp) / x.shape[0]
    return tmp**2

def std_dev(x, t, w):
    return np.sqrt(variance(x, t, w))

def variance(x, t, w):
    return MSEloss(x, t, w)

def get_likelihood(x, t, w):
    likelihood = 1
    standard_deviation = std_dev(x, t, w)
    for i in range(len(list(t))):
        X_i = np.array([x[i]**n for n in range(len(w))])
        likelihood *= norm.pdf(t[i], X_i.dot(w), standard_deviation)
    return likelihood

def LOOCV(x, t, d):
    
    loss = []
    for i in range(len(x)):
        mask = np.ones(len(x), dtype=bool)
        mask[i] = False
        
        w = getOptimalParams(x[mask], t[mask], d)
        X = polynomial(x[i], d)
        loss.append((t[i] - X.dot(w))**2)
        
    return np.average(loss)

def predict(X_new, x, t, w):
    t_new = w.T.dot(X_new)
    X = polynomial(x, len(list(w))-1)
    var_new = variance(x, t, w) * X_new.T.dot(inv(X.T.dot(X))).dot(X_new)
    return [t_new, np.sqrt(var_new)]

def set_plot(ax, xlim, ylim, xticks, yticks, xlabel, ylabel):
    ax.set_xlim(xlim)
    ax.set_ylim(ylim)
    ax.set_xticks(xticks)
    ax.set_yticks(yticks)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel, rotation=0)
    
def plt_contourf(multivar_normal, w, ax, size=5):
    x, y = np.mgrid[w[0]-size:w[0]+size:.005, w[1]-size:w[1]+size:.005]
    pos = np.empty(x.shape + (2,))
    pos[:, :, 0] = x; pos[:, :, 1] = y
    ax.contourf(x, y, multivar_normal.pdf(pos), cmap=plt.get_cmap('gist_heat'))#matplotlib.cm.winter)