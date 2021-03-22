import numpy as np

from proj1_helpers import sigmoid 

def compute_loss(y, tx, w):
    """Compute the loss by mean squared error for linear regression."""
    e = y - tx.dot(w)
    mse = np.sum(e**2) / (2* len(e))
    return mse

def logistic_loss(y, tx, w):
    """Compute the loss by negative log likelihood for the logistic regression."""
    epsilon = 1e-15
    num_samples = len(y)
    pred = sigmoid(tx.dot(w))
    pred = np.clip(pred, epsilon, 1-epsilon)
    loss = y.T.dot(np.log(pred)) + (1 - y).T.dot(np.log(1 - pred))
    return np.squeeze(- loss)

def reg_logistic_loss(y, tx, w,lambda_):
    """Compute the regularized logistic loss by negative log likelihood."""
    loss = logistic_loss(y, tx, w) + lambda_ * np.squeeze(w.T.dot(w))
    return loss