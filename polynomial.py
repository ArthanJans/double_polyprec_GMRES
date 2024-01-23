import numpy as np
from scipy.linalg import eigvals
from scipy.sparse.linalg import eigs
from scipy import linalg
from scipy.io import loadmat
from scipy import sparse
import math
from arnoldi import arnoldi

#import matplotlib.pyplot as plt


def polynomial(A: np.ndarray, x0: np.ndarray, b: np.ndarray, dim: int):
    n = A.shape[0] # Size of matrix A
    r0 = b - (A @ x0) # Starting residual

    beta = np.linalg.norm(r0, 2) # Norm of starting residual

    V = np.zeros((n,dim+1),dtype=complex, order='F') # Matrix of row vectors that form a basis of the krylov subspace of A
    H = np.zeros((dim+1, dim),dtype=complex, order='F') # Hessenberg matrix H such that A @ V[:n] = V[:n+1] @ H[:n]
    V[:,0] = r0/beta # First vector of the krylov subspace's basis is the normalised starting residual

    for j in range(dim):
        # Arnoldi
        V[:,j+1] = arnoldi(A, V, j, H[:j+2,j]) # Add a new vector to the basis in V and add a new row to H
        if H[j+1,j] == 0: # If the new vector was eliminated during orthoganisation, stop iterating
            break

    # IMPORTANT : do thorough check of the Arnoldi relation before continuing
    lhsArn = A @ (V[:,0:dim])
    rhsArn = V @ H
    print("Correctness of Arnoldi relation : ", np.linalg.norm(lhsArn-rhsArn,'fro')/np.linalg.norm(lhsArn,'fro'))
    # and also of orthonormality of the basis
    lhsOrth = np.empty([dim+1,dim+1],dtype=complex)
    for ix in range(dim+1):
        for jx in range(dim+1):
            lhsOrth[ix,jx] = np.vdot(V[:,ix],V[:,jx])
    rhsOrth = np.identity(V.shape[1])
    print("Orthonormality of Arnoldi basis : "+str(np.linalg.norm(lhsOrth-rhsOrth,'fro')/np.linalg.norm(rhsOrth,'fro')))

    return eigvals(H.transpose()[:j+1,:j+1])

def eval_poly(A: np.ndarray, v: np.ndarray, ritz: np.ndarray):
    n = A.shape[0]
    d = ritz.shape[0]
    prod = v
    p = np.zeros((n, 1))
    i = 0
    while i < d-1:
        if(np.imag(ritz[i]) == 0):
            p = p+(1/ritz[i])*prod
            prod = prod - (1/ritz[i])*A@prod
            i+=1
        else:
            a = np.real(ritz[i])
            b = np.imag(ritz[i])
            tmp = 2 * a * prod - A @ prod
            p = p + (1/(a*a+b*b)) * tmp
            if i < d-2:
                prod = prod - (1/(a*a+b*b)) * A @ tmp
            i+=2
        
    if np.imag(ritz[d-1]) == 0:
        p = p + (1/ritz[d-1]) * prod

    return p


if __name__ == "__main__":
    A = loadmat("4x4x4x4b6.0000id3n1.mat")['D']
    n = A.shape[0]

    #plt.spy(A[1:20,1:20])
    #plt.show()

    # compute the smallest (in magnitude) eigenvalues of A
    smallEvalsA,smallEvecsA = eigs(A,50,which="SM")
    print("Smallest eigenvalues : "+str(smallEvalsA))

    # eig,_ = sparse.linalg.eigs(A, k=30)
    # print(eig)

    np.random.seed(54321)
    x0 = np.zeros(n)
    b = np.random.uniform(size=n)
    d = 100
    ritz = polynomial(A, x0, b, d)
    # print("Ritz values : "+str(1./ritz))

    v = np.random.rand(n)
    vnorm = np.linalg.norm(v)
    # print(ritz)
    pv = eval_poly(A, A*v, ritz)
    print(np.linalg.norm(pv - v)/vnorm)
