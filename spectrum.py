import numpy as np
import time
from scipy.io import loadmat
from numpy.linalg import eigvals
from polynomial import polynomial, eval_poly
import sys
import matplotlib.pyplot as plt

tot = 0

def callback(args):
    global tot
    tot += 1

if __name__ == "__main__":
    A = loadmat("4x4x4x4b6.0000id3n1.mat")['D']
    n = A.shape[0]
    A = A.toarray()
    
    
    x0 = np.zeros(n)
    b = np.ones(n) 
    start = time.perf_counter()

    np.random.seed(54321)
    x0 = np.zeros(n)
    b = np.random.uniform(size=n)
    d = int(sys.argv[1])
    ritz = polynomial(A, x0, b, d)

    identity = np.eye(n)

    poly = eval_poly(A, identity, ritz)
    apoly = A @ poly
    apolyeigvals = eigvals(apoly)

    # aeigvals = eigvals(A)
    # # extract real part 
    # x = [ele.real for ele in aeigvals] 
    # # extract imaginary part 
    # y = [ele.imag for ele in aeigvals]
    # plt.scatter(x, y, s=5) 
    # plt.ylabel('Imaginary') 
    # plt.xlabel('Real') 
    # plt.title('SPECTRUM OF A')
    # plt.savefig('A_spectrum.png')
    # plt.clf()

    # extract real part 
    x = [ele.real for ele in apolyeigvals] 
    # extract imaginary part 
    y = [ele.imag for ele in apolyeigvals]
    fig,ax = plt.subplots()
    ax.plot(x, y, marker='.', linestyle='') 
    ax.ticklabel_format(axis='both', useOffset=False)
    plt.ylabel('Imaginary') 
    plt.xlabel('Real') 
    plt.title('SPECTRUM OF Ap(A) for d=' + str(d))
    plt.savefig('Ap(A)_d_' + str(d) + '_spectrum.png')
