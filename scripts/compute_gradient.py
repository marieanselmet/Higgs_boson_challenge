import numpy as np
from proj1_helpers import *

def compute_gradient(y, tx, w):
    """Computes the gradient for the linear gradient descent"""
    num_samples = len(y)
    error = y-np.dot(tx,w)
    grad_w = (-1/num_samples)*np.dot(tx.transpose(),error)
    return error, grad_w

def logistic_gradient(y, tx, w):
    """Computes the gradient for the logistic gradient descent."""
    gradient = np.dot(tx.transpose(),(sigmoid(np.dot(tx,w))-y))
    return gradient

def reg_logistic_gradient(y, tx, w, lambda_):
    """Computes the gradient for the regularized logistic gradient descent"""
    gradient = logistic_gradient(y, tx, w) + 2 * lambda_ * w
    return gradient
