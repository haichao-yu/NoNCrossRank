from scipy import sparse


def j_cr(Anorm, Ynorm, I_n, r, e, alpha, c):
    """
    Cross Rank objective function value
    
    :param Anorm: the aggregated normalized adjacency matrix of domain-specific networks
    :param Ynorm: the normalized matrix encoding the cross-domain mapping information
    :param I_n: an identity matrix of size n x n
    :param r: the ranking vector
    :param e: the query vector
    :param alpha: a regularization parameter for cross-network consistency
    :param c: a regularization parameter for query preference
    
    :return: objective value
    """

    X = I_n - Ynorm

    Obj = c * r.transpose().dot(I_n - Anorm).dot(r)[0, 0] + (1 - c) * sparse.linalg.norm(r - e) ** 2 + 2 * alpha * (r.transpose().dot(X).dot(r))[0, 0]

    return Obj