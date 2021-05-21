def modifiedGramSchmidt(A):
    """
    Gives a orthonormal matrix, using modified Gram Schmidt Procedure
    :param A: a matrix of column vectors
    :return: a matrix of orthonormal column vectors

    found here: https://stackoverflow.com/questions/47349233/modified-gram-schmidt-in-python-for-complex-vectors
    """
    # assuming A is a square matrix
    dim = A.shape[0]
    Q = np.zeros(A.shape, dtype=A.dtype)
    for j in range(0, dim):
        q = A[:,j]
        for i in range(0, j):
            rij = np.vdot(Q[:,i], q)
            q = q - rij*Q[:,i]
        rjj = np.linalg.norm(q, ord=2)
        if np.isclose(rjj,0.0):
            raise ValueError("invalid input matrix")
        else:
            Q[:,j] = q/rjj
    return Q


def gram_schmidt(A):
	""" Representation of Gram-Schmidt Process or QR Diagonalization 
		for an mxn system of linear equations. 
        https://github.com/philwilt/gram_schmidt/blob/master/gram_schmidt.py
    """
		
	m = np.shape(A)[0]
	n = np.shape(A)[1]

	Q =  np.zeros((m, m))
	R =  np.zeros((n, n)) 

	for j in xrange(n):
		
		v = A[:,j]
		
		for i in xrange(j):
			
			R[i,j] = Q[:,i].T * A[:,j]

			v = v.squeeze() - (R[i,j] * Q[:,i])

		R[j,j] =  np.linalg.norm(v)
		Q[:,j] = (v / R[j,j]).squeeze()

	return Q, R
