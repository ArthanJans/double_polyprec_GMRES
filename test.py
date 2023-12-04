from gmres import gmres
import numpy as np
import time
import scipy.sparse.linalg as spla

if __name__ == "__main__":
    n = 101
    # A = sp.rand(n,n,0.01,random_state=1,format="csr")
    # np.random.seed(1)
    # A = np.random.rand(n,n)


    # A = [2, 1, 0, 0, ..., 0]
    #     [1, 2, 1, 0, ..., 0]
    #     [..................]
    #     [0, ..., 0, 1, 2, 1]
    #     [0, ..., 0, 0, 1, 2]
    A = np.zeros((n,n))
    for i in range(n):
        A[i,i] = 2
        if i > 0:
            A[i-1,i] = 1
            A[i,i-1] = 1
    
    x0 = np.zeros(n)
    b = np.ones(n) 
    start = time.perf_counter()
    x = gmres(A, x0, b, n, 0.00001)
    print("My time:", time.perf_counter() - start)
    

    # print(np.allclose(A.dot(x), b))
    # print("My x:",x)
    # print("My Ax:",A @ x)
    print("My residual:", np.linalg.norm(b - (A @ x), 2))
    start = time.perf_counter()
    x,_ = spla.gmres(A,b,x0,restart=2*n)
    print("Scipy time:", time.perf_counter() - start)
    
    # print(np.allclose(A.dot(x), b))
    # print("Scipy x:",x)
    # print("Scipy Ax:",A @ x)
    print("Scipy residual:", np.linalg.norm(b-(A@x),2))