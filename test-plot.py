"""
A simple test program to verify matplot is correctly installed.
"""

__author__ = "Daniel Chapin"

import numpy as np
from matplotlib import pyplot as plt


def main():
    xvals = np.linspace(0, 34.5, 10)
    yvals = xvals * 2

    plt.plot(xvals, yvals)
    plt.xlabel("Time in Math 251 (seconds)")
    plt.ylabel("Level of insanity (confusion squared)")
    plt.title("Math 251 Project")
    plt.show()

if __name__ == "__main__":
    main()
