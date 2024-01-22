import numpy as np
from scipy.linalg import eigvals
from scipy import linalg
from scipy.io import loadmat
from scipy import sparse
import math
from arnoldi import arnoldi


def polynomial(A: np.ndarray, x0: np.ndarray, b: np.ndarray, dim: int):
    m = A.shape[0] # Size of matrix A
    r0 = b - (A @ x0) # Starting residual

    beta = np.linalg.norm(r0, 2) # Norm of starting residual

    # Following 2 matrices are stored as transpose of what is usually seen in literature. This is for performance since numpy is quicker when chaning the last index, instead of the first.
    V = np.zeros((dim+1,m)) # Matrix of row vectors that form a basis of the krylov subspace of A
    H = np.zeros((dim, dim+1)) # Hessenberg matrix H such that A @ V[:n] = V[:n+1] @ H[:n]
    V[0] = r0/beta # First vector of the krylov subspace's basis is the normalised starting residual

    for j in range(dim):
        # Arnoldi
        V[j+1], H[j,:j+2] = arnoldi(A, V, j) # Add a new vector to the basis in V and add a new row to H
        if H[j,j+1] == 0: # If the new vector was eliminated during orthoganisation, stop iterating
            break

    return eigvals(H[:j+1,:j+1])

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
    A = loadmat("ted_B.mat")["Problem"]["A"][0][0]
    n = A.shape[0]
    print(n)
    # eig,_ = sparse.linalg.eigs(A, k=30)
    # print(eig)
    x0 = np.zeros(n)
    b = np.ones(n) 
    np.random.seed(1)
    v = np.random.rand(n)
    vnorm = np.linalg.norm(v)
    d=5
    ritz = polynomial(A, x0, b, d)
    print(ritz)
    # print(ritz)
    # eigdiff = eig
    # eigdiff[:d] -= ritz
    # print(np.linalg.norm(eigdiff))
    pv = eval_poly(A, A*v, ritz)
    print(np.linalg.norm(pv - v)/np.linalg.norm(v))

