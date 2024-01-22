import numpy as np
import math

from arnoldi import arnoldi

"""
Solve equation Ax = b using gmres algorithm
Arguments:
A             :matrix being solved
x0            :initial guess for x0
b             :desired result of Ax
maxIterations :maximum iterations of GMRES
threshold     :threshold for residual, stops iterations if residual is below threshold

Returns:
x             :solution to Ax = b
"""
def gmres(A: np.ndarray, x0: np.ndarray, b: np.ndarray, maxIterations: int, threshold: float):
    m = A.shape[0] # Size of matrix A
    r0 = b - (A @ x0) # Starting residual

    beta = np.linalg.norm(r0, 2) # Norm of starting residual
    bNorm = np.linalg.norm(b, 2) # Norm of b, used to calculate error

    # Following 2 matrices are stored as transpose of what is usually seen in literature. This is for performance since numpy is quicker when chaning the last index, instead of the first.
    V = np.zeros((maxIterations+1,m)) # Matrix of row vectors that form a basis of the krylov subspace of A
    H = np.zeros((maxIterations,maxIterations+1)) # Hessenberg matrix H such that A @ V[:n] = V[:n+1] @ H[:n]

    V[0] = r0/beta # First vector of the krylov subspace's basis is the normalised starting residual

    sn = np.zeros(maxIterations) # Store the sines for givens rotations
    cs = np.zeros(maxIterations) # Store the cosines for givens rotations

    betaE = np.zeros(maxIterations+1) # Vector of [beta, 0, ..., 0] Used for calculating the residual at each iteration
    betaE[0] = beta

    # Perform a maximum of maxIterations iterations of GMRES
    for j in range(maxIterations):
        # Arnoldi
        V[j+1], H[j,:j+2] = arnoldi(A, V, j) # Add a new vector to the basis in V and add a new row to H
        if H[j,j+1] == 0: # If the new vector was eliminated during orthoganisation, stop iterating
            break

        # Apply Givens Rotation
        sn[j], cs[j] = givens(H[j], j, sn, cs) # Apply all previous givens rotations to the new row of H and apply a new rotation to turn it into a square lower trianguar matrix (transpose is upper triangular)

        # Apply the same givens rotation to betaE
        betaE[j+1] = -1 * sn[j] * betaE[j] 
        betaE[j] = cs[j] * betaE[j] 

        # j+1th element in betaE is the residual after this iteration
        # residual over norm of b
        error = abs(betaE[j+1])/bNorm
        if error <= threshold: # If relative error is low enough, stop iterations
            break

    y = np.zeros(j+1) # y = miny(H @ y - betaE) => H @ y = betaE => y = inv(H) @ betaE
    for i in range(j, -1, -1): # Solve for y using back substitution
        y[i] = ((betaE[i] - np.sum(H[i+1:j+1, i] * y[i+1:])) / H[i,i])

    x = x0 + (np.transpose(V[:j+1]) @ y) # x is solved using V and y
    return x, j+1

"""
Applies givens rotations to the row of H and creates a new rotation to ensure H is square and lower triangular
Arguments:
Hj    :The jth row of the Hessenberg matrix H
j     :How many givens rotations already applied to H
sn    :The sines of the rotations
cs    :The cosines of the rotations

Returns:
snj   :The sine of the new rotation
csj   :The cosine of the new rotation
"""
def givens(Hj:np.ndarray, j: int, sn: np.ndarray, cs: np.ndarray):
    # Apply all previous givens rotations to the row Hj
    for i in range(j):
        temp = Hj[i] * cs[i] + Hj[i+1] * sn[i] 
        Hj[i+1] = Hj[i+1] * cs[i] - Hj[i] * sn[i]
        Hj[i] = temp
    
    # Apply new givens rotation to make the last element of Hj equal to 0
    divisor = math.sqrt(Hj[j]*Hj[j] + Hj[j+1] * Hj[j+1])
    snj = Hj[j+1]/divisor
    csj = Hj[j]/divisor

    Hj[j] = Hj[j] * csj + Hj[j+1] * snj
    Hj[j+1] = 0.0
    return snj, csj # Return the new rotations