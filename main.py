#!/usr/bin/env python
"""
Program to perform operations on data with propagating Monte Carlo error.

Generates random data with desired statistical properties, finds linear least-squares estimate of the original function.
Finds distribution of linear fit coefficients.
"""
__author__: "Chris Biancone, Daniel Chapin, Carissa Hartley, Isaac Kim, Neil Williamson"

import numpy as np
from matplotlib import pyplot as plt


xvals = np.linspace(-1, 1, 11)
N = 1000  # number of data sets

# straight line
def fn1(x):
    return 3 * np.ones(len(x))
# test function
def fn2(x):
    return -2.0 + 3.0 * x
# test quadratic
def fn3(x):
    return 2 * x ** 2
# unsure
def fnTrue(x):
    return 2 + 10 * x - 2 * x ** 2


def make_fake_data(fn):
    """
    Adds error to original function following std normal distribution
    :param fn: test function
    :return: test function + error
    """
    return fn(xvals) + np.random.normal(size=len(xvals))


def my_gauss(x, mu, sigma):
    """
    Calculates estimated PDF supposing a normal distribution
    :param x:
    :param mu: mean
    :param sigma: standard deviation
    :return: PDF
    """
    return np.exp(-np.power(x-mu, 2) / (2.*sigma**2)) / np.sqrt(2*np.pi)


def main():
    # perform linear regression (degree = 1)
    np.polyfit(xvals, make_fake_data(fn2), 1)
    # alpha = np.dot((np.dot(np.linalg.inv(np.dot(A.T, A)), A.T)), y)  # manual pseudo-inverse for comparison
    dat_out = []  # array for coefficients of each data set
    for i in np.arange(N):  # iterate over 1000 data sets
        dat_out.append(np.polyfit(xvals, make_fake_data(fn2), 1))
    dat_out = np.array(dat_out)
    # print(dat_out)
    c1 = dat_out[:, 0]
    c1mean = np.mean(dat_out[:, 0])
    c1std = np.std(dat_out[:, 0])
    c1var = np.var(dat_out[:, 0])
    c0 = dat_out[:, 1]
    c0mean = np.mean(dat_out[:, 1])
    c0std = np.std(dat_out[:, 1])
    c0var = np.var(dat_out[:, 1])
    covMatrix = np.cov(dat_out[:, 0], dat_out[:, 1])

    print("1st degree coefficient:\n\tmean:", c1mean, "\n\tstd dev:", c1std, "\n\tvariance:", c1var)
    print("0th degree coefficient:\n\tmean:", c0mean, "\n\tstd dev:", c0std, "\n\tvariance:", c0var)
    print("covariance matrix:\n", covMatrix)

    # get stuff ready for cdf
    c1sort = np.sort(c1)
    c0sort = np.sort(c0)
    cdf1 = np.arange(N) / float(N)
    cdf0 = np.arange(N) / float(N)

    # plot coefficient spread
    plt.figure(1)
    plt.title("Coefficient Spread")
    plt.xlabel(r'$\hat{c}_1$')
    plt.ylabel(r'$\hat{c}_0$')
    plt.scatter(dat_out[:, 0], dat_out[:, 1])
    plt.show()

    # plot cdfs (for reference)
    plt.figure(2)
    plt.title(r'$\hat{c}_1$ CDF')
    plt.xlabel(r'$\hat{c}_1$')
    plt.ylabel('P')
    plt.plot(c1sort, cdf1)
    plt.figure(3)
    plt.title(r'$\hat{c}_0$ CDF')
    plt.xlabel(r'$\hat{c}_0$')
    plt.ylabel('P')
    plt.plot(c0sort, cdf0)
    plt.show()

    # estimated distribution - unsure about this yet
    c1PDF = my_gauss(xvals, c1mean, c1std)
    plt.figure(4)
    plt.plot(xvals, c1PDF)
    # plt.savefig("my_fig.png")
    plt.title("Estimated Distribution of Data")
    plt.ylabel(r'${\cal L}_m^\gamma$')
    plt.show()


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    main()
