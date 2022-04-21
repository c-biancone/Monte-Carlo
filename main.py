#!/usr/bin/env python
"""
Program to perform operations on data with propagating Monte Carlo error.

Generates random data with desired statistical properties, finds linear least-squares estimate of the original function.
Finds distribution of linear fit coefficients.
"""
__author__: "Chris Biancone, Daniel Chapin, Carissa Hartley, Isaac Kim, Neil Williamson, Dylan Daccarett"

import numpy as np
from matplotlib import pyplot as plt
from datparser import parse_dat


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


def gaussian(x, mu, sigma):
    """
    Calculates estimated PDF supposing a normal distribution
    :param x:
    :param mu: mean
    :param sigma: standard deviation
    :return: PDF
    """
    return np.exp(-np.power(x-mu, 2) / (2*sigma**2)) / np.sqrt(2*np.pi)


def ecdf(a):
    """
    Calculates empirical CDF for an arbitrary array of data. Essentially a fancy sum.
    :param a: input array
    :return: CDF with points in sorted order
    """
    x, counts = np.unique(a, return_counts=True)
    cusum = np.cumsum(counts)
    x = np.insert(x, 0, x[0])
    y = cusum / cusum[-1]
    y = np.insert(y, 0, 0.)
    return x, y


def main():
    dat_out = []  # array for coefficients of each data set
    y_out = []
    origfcn = make_fake_data(fn2)
    for i in np.arange(N):  # iterate over 1000 data sets
        # perform regression using Ordinary Least Squares algorithm
        # degree = 1 for linear regression
        yvals = make_fake_data(fn2)
        dat_out.append(np.polyfit(xvals, yvals, 1))
        y_out.append(yvals)
    dat_out = np.array(dat_out)
    y_out = np.array(y_out)  # array of random generated data for computing variance
    # print(dat_out)
    c1 = dat_out[:, 0]
    c1mean = np.mean(dat_out[:, 0])
    c1std = np.std(dat_out[:, 0])
    c1var = np.var(dat_out[:, 0])
    c0 = dat_out[:, 1]
    c0mean = np.mean(dat_out[:, 1])
    c0std = np.std(dat_out[:, 1])
    c0var = np.var(dat_out[:, 1])
    covMatrix = np.cov(dat_out[:, 0], dat_out[:, 1])  # [[sig_c1, cov],[cov, sig_c0]]
    correlation = covMatrix[1, 1] / c1std * c0std  # cov(c0, c1) / sig_c1 * sig_c0
    # seems independent of function choice

    # sample variance
    x1var = np.var(y_out[:, 0])
    print("Sample variance:", x1var)
    x2var = np.var(y_out[:, 1])
    print("Sample variance:", x2var)


    print("Generated data =================================")
    print("1st degree coefficient:\n\tmean:", c1mean, "\n\tstd dev:", c1std, "\n\tvariance:", c1var)
    print("0th degree coefficient:\n\tmean:", c0mean, "\n\tstd dev:", c0std, "\n\tvariance:", c0var)
    print("covariance matrix:\n", covMatrix)
    print("correlation coefficient:", correlation)

    # plot coefficient spread
    plt.figure(1)
    plt.title("Coefficient Spread, N = " + str(N))
    plt.xlabel(r'$\hat{c}_1$')
    plt.ylabel(r'$\hat{c}_0$')
    plt.scatter(dat_out[:, 0], dat_out[:, 1])
    plt.savefig("coefficient_spread")
    plt.show()

    # estimated distribution and empirical CDF
    c1pdf = gaussian(c1, c1mean, c1std)
    c1x, c1cdf = ecdf(c1)
    plt.figure(2)
    fig, ax1 = plt.subplots()
    ax2 = ax1.twinx()
    ax1.scatter(c1, c1pdf, s=2)
    ax2.plot(c1x, c1cdf, 'r')
    ax1.set_xlabel(r'$\hat{c}_1$')
    ax1.set_ylabel("PDF", color='b')
    ax2.set_ylabel("CDF", color='r')
    plt.title(r'Estimated Distribution of $\hat{c}_1$')
    plt.savefig("c1hat_dist")
    plt.show()

    c0pdf = gaussian(c0, c0mean, c0std)
    c0x, c0cdf = ecdf(c0)
    plt.figure(3)
    fig, ax1 = plt.subplots()
    ax2 = ax1.twinx()
    ax1.scatter(c0, c0pdf, s=2)
    ax2.plot(c0x, c0cdf, 'r')
    ax1.set_xlabel(r'$\hat{c}_0$')
    ax1.set_ylabel("PDF", color='b')
    ax2.set_ylabel("CDF", color='r')
    plt.title(r'Estimated Distribution of $\hat{c}_0$')
    plt.savefig("c0hat_dist")
    plt.show()

    # plot original data and estimated curve
    plt.figure(4)
    regression = c1mean * xvals + c0mean
    plt.scatter(xvals, origfcn, label='Original Function')
    plt.plot(xvals, regression, 'r', label='Linear Regression')
    plt.title("Function and Estimate")
    plt.legend()
    plt.savefig("estimate_random-data")
    plt.show()

    # variance of f_hat:
    # Var(aX + b) = a^2(Var(X)) - this right??
    varfhat = np.var(c1mean * xvals + c0mean)
    print("Variance of estimated function:", varfhat)

    # ==================================================================================================================
    # using data provided

    print("Given Data =========================")
    xreal, yreal = parse_dat('classA0.dat')  # parse data
    dat_out = [np.polyfit(xreal, yreal, 1)]
    dat_out = np.array(dat_out)
    print(dat_out)
    c1 = dat_out[:, 0]
    c1mean = np.mean(dat_out[:, 0])
    c1std = np.std(dat_out[:, 0])
    c1var = np.var(dat_out[:, 0])
    c0 = dat_out[:, 1]
    c0mean = np.mean(dat_out[:, 1])
    c0std = np.std(dat_out[:, 1])
    c0var = np.var(dat_out[:, 1])
    # covMatrix = np.cov(dat_out[:, 0], dat_out[:, 1])  # [[sig_c1, cov],[cov, sig_c0]]
    # correlation = covMatrix[1, 1] / covMatrix[0, 0] * covMatrix[1, 1]  # cov(c0, c1) / sig_c1 * sig_c0

    print("1st degree coefficient:\n\tmean:", c1mean, "\n\tstd dev:", c1std, "\n\tvariance:", c1var)
    print("0th degree coefficient:\n\tmean:", c0mean, "\n\tstd dev:", c0std, "\n\tvariance:", c0var)
    print("correlation coefficient (from before):", correlation)
    # varfhatreal = np.var(c1mean * xreal + c0mean)
    # print("Variance of estimated function:", varfhatreal)

    # 90% confidence interval: f_hat +/- z_.05sqrt(Var(f_hat)), Var(f_hat) = c_1^2
    freal = c1mean * np.array(xreal) + c0mean  # needed to convert to numpy array from python list
    confidence = 1.645*np.sqrt(varfhat)  # using variance of f_hat from generated data bc no way to get sample variance\
    print("CI: f_hat +/-", confidence)
    # from given data
    plt.figure(5)
    plt.title('Given data and regression with 90% CI')
    plt.plot(xreal, freal, 'r', label='Linear Regression')  # estimated function
    plt.fill_between(xreal, freal + confidence, freal - confidence, facecolor='red', alpha=0.5)  # confidence interval using previous variance
    plt.scatter(xreal, yreal, label='Given Data')  # given data points
    plt.savefig("estimate_given-data")
    plt.show()


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    main()
