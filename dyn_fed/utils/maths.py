
"""Maths utilities
"""
import numpy as np

def sigmoid(s):
    """
    Sigmoid calculation for the matrix passed into this function
    """
    # print(s)
    return 1./(1. + np.exp(-s))

def linspace_quantization(matrix, interval=8):
    """Quantizes parameters

    Arguments:
        W (np.ndarray): Parameter matrix to be quantized

    Returns:
        msg (np.ndarray): Structured numpy array that is quantized

    """
    min_W_val = matrix.min() + 1e-8
    max_W_val = matrix.max() + 1e-8
    bins = np.linspace(min_W_val, max_W_val, interval)
    W_bins = np.digitize(matrix, bins).astype(np.int8)

    struct_field_names = ["min_val", "max_val", "interval", "bins"]
    struct_field_types = [np.float32, np.float32, np.int32, 'b']
    struct_field_shapes = [1, 1, 1, (matrix.shape)]

    msg = np.zeros(
        1,
        dtype=(list(zip(struct_field_names, struct_field_types, struct_field_shapes)))
    )
    msg[0] = (min_W_val, max_W_val, interval, W_bins)

    return msg

def reconstruct_approximation(buf, shape, r_dtype=np.float32):
    """Reconstruct linear space appromixation

    Args:
        buf: Buffer to be reconstructed
        shape (tuple): Shape to be reconstructed to
        r_dtype (np.type): Type of reconstructed array

    Returns:
        matrix (numpy.ndarray): Reconstructed array
    """
    # Reconstruct W matrix from min, max, no. of intervals and which bins
    # each parameter value falls in
    struct_field_names = ["min_val", "max_val", "interval", "bins"]
    struct_field_types = [np.float32, np.float32, np.int32, 'b']
    struct_field_shapes = [1, 1, 1, (shape)]
    dtype = (list(zip(struct_field_names, struct_field_types, struct_field_shapes)))
                                    
    data = np.frombuffer(buf, dtype=dtype)
    min_W_val, max_W_val, interval, matrix_bins = data[0]

    # Generate lineared space vector
    bins = np.linspace(min_W_val, max_W_val, interval, dtype=r_dtype)
    matrix = bins[matrix_bins].reshape(shape)

    return matrix

def arg_svd(X: np.ndarray, percentile=0.95):
    """Returns index of feature where there is 95% of the variance of the
    data

    Returns:
        idx_95: (Sum of singular values of 95 percentile)
    """
    # Only interested in the singular values
    X = X.copy()
    if X.ndim > 2:
        X = X.reshape(X.shape[0], -1) # Reshape to 2d
    _, s, _ = np.linalg.svd(X, full_matrices=False)

    idx_95 = np.argwhere(np.cumsum(s) / np.sum(s) >= percentile).flatten()[0]

    return idx_95
