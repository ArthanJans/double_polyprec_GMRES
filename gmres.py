import numpy as np
import math

def gmres(A: np.ndarray, x0: np.ndarray, b: np.ndarray, maxIterations: int, threshold: float):
    m = A.shape[0]
    r0 = b - (A @ x0)
    beta = np.linalg.norm(r0, 2)
    bNorm = np.linalg.norm(b, 2)
    V = np.zeros((maxIterations+1,m))
    H = np.zeros((maxIterations,maxIterations+1))
    V[0] = r0/beta
    sn = np.zeros(maxIterations)
    cs = np.zeros(maxIterations)
    betaE = np.zeros(maxIterations+1)
    betaE[0] = beta
    arnoldis = 0
    applyGivens = 0
    iterations = 0
    for j in range(maxIterations):
        # Arnoldi
        wj = (A @ V[j])
        for i in range(j+1):
            H[j,i] = (wj @ V[i])
            wj -= (H[j,i] * V[i])
            iterations += 1
        H[j,j+1] = np.linalg.norm(wj, 2)
        if H[j,j+1] == 0:
            break
        V[j+1] = wj/H[j,j+1]

        # Apply Givens Rotation
        for i in range(j):
            temp = H[j,i] * cs[i] + H[j,i+1] * sn[i]
            H[j,i+1] = H[j,i+1] * cs[i] - H[j,i] * sn[i]
            H[j,i] = temp
        divisor = math.sqrt(H[j,j]*H[j,j] + H[j,j+1] * H[j,j+1])
        sn[j] = H[j,j+1]/divisor
        cs[j] = H[j,j]/divisor
        H[j,j] = H[j,j] * cs[j] + H[j,j+1] * sn[j]
        H[j,j+1] = 0.0
        betaE[j+1] = -1 * sn[j] * betaE[j]
        betaE[j] = cs[j] * betaE[j]
        # residual over norm of b
        error = abs(betaE[j+1])/bNorm
        if error <= threshold:
            break
    y = np.zeros(j+1)
    for i in range(j, -1, -1):
        y[i] = ((betaE[i] - np.sum(H[i+1:j+1, i] * y[i+1:])) / H[i,i])
    x = x0 + (np.transpose(V[:j+1]) @ y)
    return x