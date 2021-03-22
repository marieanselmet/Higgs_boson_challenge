import numpy as np
from data_helpers import *
from compute_gradient import *
from cost import *
from proj1_helpers import *

def least_squares_GD(y, tx, initial_w, max_iters, gamma, printing=True):
    """Linear regression using gradient descent for max_iters iteration given
    the input labelled data y, tx with initial_w and gamma as the initial weight and the
    learning rate respectively. Return final weights and loss"""
    
    w = initial_w
    losses = []
    
    # Set a threshold to stop the iterations before max_iters if a great approximation of the optimum is obtained
    thres = 1e-8 
    
    # Gradient descent iterations
    for n_iter in range(max_iters):
        
        # Compute gradient, loss and update w
        _,grad = compute_gradient(y,tx,w)
        loss = compute_loss(y,tx,w)
        w = w - gamma*grad
        
        losses.append(loss)
        
        
        if len(losses) > 1 and np.abs(losses[-1]-losses[-2]) < thres:
            """If the difference between the two last computed losses becomes inferior to thres, 
             we can estimate that the gradient of the loss approaches zero and that we tend to the optimum"""
            break
            
        if printing==True:    
            print("Gradient Descent({bi}/{ti}): loss={l}, weights = {we}".format(
                bi=n_iter, ti=max_iters - 1, l=loss, we = w ))
        
    return w, losses[-1]

def least_squares_SGD(y, tx, initial_w, batch_size, max_iters, gamma, printing=True):
    """Linear regression using stochastic gradient descent for max_iters iteration given
    the input labelled data y, tx with initial_w and gamma as the initial weight and the
    learning rate respectively. Return final weights and loss"""
    
    w = initial_w
    
    # SGD iterations on batches of batch_size size
    for n_iter in range(max_iters):
        for minibatch_y, minibatch_tx in batch_iter(y, tx, batch_size):
        
            # Compute gradient, loss and update w
            _, grad = compute_gradient(minibatch_y,minibatch_tx,w)
            loss = compute_loss(minibatch_y,minibatch_tx,w)
            w = w - gamma*grad
        
        if printing==True: 
            print("SGD({bi}/{ti}): loss={l}, weights={w}".format(
                bi=n_iter, ti=max_iters - 1, l=loss, w=w))   
    
    return w, loss

def least_squares(y, tx):
    """Calculate optimal w and loss by least squares given input labelled
    data y and tx. Return weights and loss."""
    n = len(y)
    w = np.linalg.solve (np.dot(tx.transpose(),tx),np.dot(tx.transpose(),y))
    loss = compute_loss(y, tx, w)
    
    return w, loss

def ridge_regression(y, tx, lambda_):
    """Calculate optimal w and loss by ridge regression given input labelled
    data tx and y. Return weights and loss."""
    n = len(y)
    a = np.dot(tx.transpose(),tx)+(2*n)*lambda_*np.identity(tx.shape[1])
    b = np.dot(tx.transpose(),y)
    w =np.linalg.solve(a, b)

    loss = compute_loss(y, tx, w)
    
    return w, loss

def logistic_regression(y, tx, initial_w, max_iters, gamma):
    """Logistic regression using gradient descent for max_iters iteration given
    the input labelled data y, tx with initial_w and gamma as the initial weight and the
    learning rate respectively. Return final weights and loss"""
    losses = []
    w = initial_w
    num_samples = len(y)
    
    # Gradient descent iterations
    for iter in range(max_iters):
        sum_loss = 0
        
        for batch_y, batch_tx in batch_iter(y, tx, batch_size=1, num_batches = 1):
            
            # Compute gradient and update weight
            gradient = logistic_gradient (batch_y,batch_tx,w)
            w -= gamma*gradient
            
            # Compute loss
            loss = logistic_loss (batch_y,batch_tx,w)
            
            losses.append(loss)

    loss = logistic_loss(y,tx,w)
    
    return w, loss

def reg_logistic_regression(y, tx, lambda_, initial_w, max_iter, gamma): 
    """Regularized logistic regression using gradient descent for max_iters iteration given
    the input labelled data y, tx with initial_w, gamma and lambda as the initial weight,
    learning rate and regularization factor respectively. Return final weights and loss"""
    
    num_samples = len(y)
    losses = []
    w = initial_w
    
    for iter in range(max_iter):
       
        for batch_y, batch_tx in batch_iter(y, tx, batch_size=1, num_batches = 1):
            
            gradient = reg_logistic_gradient(batch_y,batch_tx,w, lambda_)
            w -= gamma*gradient
            loss = reg_logistic_loss(batch_y,batch_tx,w,lambda_)
            
            
        losses.append(loss)
        
    # Calculate loss over the whole training set with L2-regularization
    loss = reg_logistic_loss(y,tx,w,lambda_)
    
    return w, loss