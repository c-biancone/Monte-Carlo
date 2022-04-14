
#Authors: Carissa Hartley

import numpy as np
from matplotlib import pyplot as plt

xvals = np.linspace(-1,1, 11)
fn1 = lambda x: 3*np.ones(len(xvals))
fn2 = lambda x: -2+3.*x
fn3 = lambda x: 2*x**2
fnTrue =lambda x: 2+10*x-2*x**2

def makeFakeData(fn):
       return fn(xvals)+np.random.normal(size=len(xvals))

def main():
    np.polyfit(xvals,makeFakeData(fn1),2)
    dat_out = []
    for indx in np.arange(500):
        dat_out.append(np.polyfit(xvals,makeFakeData(fn3),2))
    dat_out = np.array(dat_out)
    print(dat_out)
    print(np.mean(dat_out[:,0]))
    print(np.mean(dat_out[:,1]))
    print(np.std(dat_out[:,0]))
    print(np.cov(dat_out[:,0],dat_out[:,1]))

    plt.scatter(dat_out[:,0],dat_out[:,1])


    #def mygauss(x, mu, sigma):
    #    np.exp(- np.power(x-mu,2)/(2.*sigma**2) ) /np.sqrt(2*np.pi)
    xvalsTest = np.linspace(-20,20,500)
    #yvalsTest =  mygauss(xvalsTest,0,1)
    yvalsTest = np.exp( - np.power(xvalsTest-0,2)/(2.*1**2) )/ np.sqrt(2*np.pi)
    plt.plot(xvalsTest,yvalsTest)
    plt.savefig("my_fig.png")
    plt.xlabel("Hello there")
    plt.ylabel(r'${\cal L}_m^\gamma$')
    plt.show()

    



# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    main()

# See PyCharm help at https://www.jetbrains.com/help/pycharm/
