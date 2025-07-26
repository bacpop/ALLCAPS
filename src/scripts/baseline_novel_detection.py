import numpy as np
from scipy.sparse import csr_matrix


def jaccard_distance_sparse(X_train: csr_matrix, X_query: csr_matrix) -> np.ndarray:
    """
    Compute the pairwise binary Jaccard distance between query and training samples

    Parameters
    ----------
    X_train : csr_matrix of shape (n_train, n_features)
        Sparse count matrix for training sequences.
    X_query : csr_matrix of shape (n_query, n_features)
        Sparse count matrix for query sequences.

    Returns
    -------
    distances : np.ndarray of shape (n_query, n_train)
        Binary Jaccard distances (1 - |A ∩ B| / |A U B|).
    """
    # Binarize: set all non-zero entries to 1
    B_train = X_train.copy()
    B_train.data[:] = 1
    B_query = X_query.copy()
    B_query.data[:] = 1

    # Intersection counts: (n_query, n_train)
    intersection = B_query.dot(B_train.T).toarray().astype(np.float64)


    # Union = |A| + |B| - |A ∩ B|
    sum_query = B_query.sum(axis=1).A1  # shape (n_query,)
    sum_train = B_train.sum(axis=1).A1  # shape (n_train,)
    union = sum_query[:, None] + sum_train[None, :] - intersection

    # Avoid division by zero (if both vectors are all-zero, define distance as 0)
    union[union == 0] = 1

    # Jaccard distance
    distances = 1.0 - (intersection / union)
    return distances
