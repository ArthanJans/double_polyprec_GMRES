import numpy as np

"""
Apply an iteration of arnoldi to create a new vector for the basis of the krylov subspace of A
Arguments:
A     :Matrix to form krylov subspace of
V     :Matrix of row vectors forming the existing basis of the krylov subspace
j     :The number of iterations of arnoldi already performed

Returns:
wj    :A new vector in the basis of the krylov subspace which is orthogonal to all other vectors in the basis and is normalised
Hj    :A new row of the Hessenberg matrix
"""
def arnoldi(A: np.ndarray, V: np.ndarray, j: int):
    wj = (A @ V[j]) # Multiply previous basis vector by A
    # FIXME : this Hj is being allocated every time that this function is
    #         called .. which might want to be changed for performance
    Hj = np.zeros(j+2,dtype=complex)

    for i in range(j+1): # Apply Gram-Schmidt to orthoganlise wj w.r.t existing basis
        #Hj[i] = (wj @ V[i])
        Hj[i] = np.vdot(wj,V[i])
        wj -= (Hj[i] * V[i])

    Hj[j+1] = np.linalg.norm(wj, 2) # Store the norm of wj and then normalise it
    return wj/Hj[j+1], Hj
