# -*- coding: utf-8 -*-
"""some helper functions for project 1."""
import csv
import numpy as np


def sigmoid(t):
    """Sigmoid function."""
    return 1/(1+np.exp(-t))

def load_csv_data(data_path, sub_sample=False):
    """Load data and return y (class labels), tX (features) and ids (event ids)"""
    y = np.genfromtxt(data_path, delimiter=",", skip_header=1, dtype=str, usecols=1)
    x = np.genfromtxt(data_path, delimiter=",", skip_header=1)
    ids = x[:, 0].astype(np.int)
    input_data = x[:, 2:]

    # Convert class labels from strings to binary (-1,1)
    yb = np.ones(len(y))
    yb[np.where(y=='b')] = -1
    
    # Sub-sample
    if sub_sample:
        yb = yb[::50]
        input_data = input_data[::50]
        ids = ids[::50]

    return yb, input_data, ids

def compute_f1_score(y, tx, w, regression = 'linear'):
    """Compute the f1 score given labelled data input data and optimal weights, 
    ie the harmonic mean of p and r where p is the number of correct positive
    results divided by the number of all positive results returned by our prediction,
    and r is the number of correct positive results divided by the number of all relevant samples.
    Can be used for linear or logistic regression. """
    
    y_pred_training = predict_labels(w, tx, regression)
    num_samples = len(y_pred_training)
    ind_pos_pred = np.ravel(np.where(y_pred_training==1))
    # Number of positive results returned by our prediction
    count_pos_training = ind_pos_pred.shape[0]
    # Number of relevant samples (ie positive results in y)
    count_relevant_pos = 0
    # Number of correct positive results
    count_true_pos = 0
                           
    if regression == 'linear':
        for i in range(num_samples):
             if (y[i] == 1):
                count_relevant_pos += 1
                if (y_pred_training[i] == 1):
                    count_true_pos += 1
    
                           
    elif regression == 'logistic': 
        for i in range(num_samples):
            if (y[i] == 1):
                count_relevant_pos += 1
                if (y_pred_training[i] == 1):
                    count_true_pos += 1         
    
    p = count_true_pos/count_pos_training
    r = count_true_pos/count_relevant_pos
    
    # Harmonic mean
    f1_score = 2/(1/p + 1/r)
    
    return f1_score

def predict_accuracy(y, tx, w, regression = 'linear'):
    """Compute the percentage of matching between predictions and actual labels
    given optimal weights w and input data y and tx. Can be used for linear and logistic
    regression. """
    y_pred_training = predict_labels(w, tx, regression)
    num_samples = len(y_pred_training)
    count = 0
    if regression == 'linear':
        for i in range(num_samples):
            if (y_pred_training[i] == y[i]):
                count += 1
    elif regression == 'logistic':
        
        for i in range(num_samples):
            if (y_pred_training[i] == -1 and y[i] == 0):
                count += 1
            if (y_pred_training[i] == 1 and y[i] == 1):
                count +=1
                
    accuracy = (count *100) / num_samples
    return accuracy

def predict_labels(weights, data, regression = 'linear'):
    """Generate class predictions given weights, and a test data matrix.
    Can be used for linear (default) and logistic regression. """
    y_pred = np.dot(data, weights)
    if regression == 'linear':
        y_pred[np.where(y_pred <= 0)] = -1
        y_pred[np.where(y_pred > 0)] = 1
    elif regression == 'logistic':
        y_pred[np.where(y_pred <= 0.5)] = -1
        y_pred[np.where(y_pred > 0.5)] = 1
        
    return y_pred


def build_k_indices(y, k_fold, seed):
    """Build k indices for k-fold. Helper functions for
    cross-validation. """
    num_row = y.shape[0]
    interval = int(num_row / k_fold) 
    # Generate random indices
    np.random.seed(seed) 
    indices = np.random.permutation(num_row)
    # Contruct set of indices for the k different folds for cross validation
    k_indices = [indices[k * interval: (k + 1) * interval]
                 for k in range(k_fold)]
    return np.array(k_indices)

def cross_validation (y, x, k, k_fold, seed):
    """Randomly partition the data into k_fold groups to train and test each sub-dataset"""
    # Split data in k_folds
    k_indices = build_k_indices(y, k_fold, seed)
    
    # Get k'th subgroup in test, others in train
    y_test = y[k_indices[k]]
    x_test = x[k_indices[k]]
    k_indices_del = np.delete(k_indices,k,0)
    y_train = y[np.ravel(k_indices_del)]
    x_train = x[np.ravel(k_indices_del)]
    
    return x_train, y_train, x_test, y_test
    

def classify (y):
    """ Convert label y from linear to logistic range"""
    for i in range(len(y)):
        if y[i] == -1:
            y[i] = 0
    return y

def batch_iter(y, tx, batch_size, num_batches=1, shuffle=True):
    """Generate a minibatch iterator for a dataset.
    Takes as input two iterables (here the output desired values 'y' and the input data 'tx')
    Outputs an iterator which gives mini-batches of `batch_size` matching elements from `y` and `tx`.
    Data can be randomly shuffled to avoid ordering in the original data messing with the randomness of the minibatches.
    Example of use :
    for minibatch_y, minibatch_tx in batch_iter(y, tx, 32):
        <DO-SOMETHING> """
    data_size = len(y)

    if shuffle:
        shuffle_indices = np.random.permutation(np.arange(data_size))
        shuffled_y = y[shuffle_indices]
        shuffled_tx = tx[shuffle_indices]
    else:
        shuffled_y = y
        shuffled_tx = tx
    for batch_num in range(num_batches):
        start_index = batch_num * batch_size
        end_index = min((batch_num + 1) * batch_size, data_size)
        if start_index != end_index:
            yield shuffled_y[start_index:end_index], shuffled_tx[start_index:end_index]


def create_csv_submission(ids, y_pred, name):
    """
    Create an output file in csv format for submission to kaggle
    Arguments: ids (event ids associated with each prediction)
               y_pred (predicted class labels)
               name (string name of .csv output file to be created)
    """
    with open(name, 'w') as csvfile:
        fieldnames = ['Id', 'Prediction']
        writer = csv.DictWriter(csvfile, delimiter=",", fieldnames=fieldnames)
        writer.writeheader()
        for r1, r2 in zip(ids, y_pred):
            writer.writerow({'Id':int(r1),'Prediction':int(r2)})
