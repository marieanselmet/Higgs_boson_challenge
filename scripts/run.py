#!/usr/bin/env python
# coding: utf-8

# Load Python packages
import numpy as np
np.warnings.filterwarnings('ignore')
import matplotlib.pyplot as plt

# Import needed functions
from proj1_helpers import *
from data_helpers import *
from implementations import ridge_regression
from cost import compute_loss



# Load train and test data
print("Loading train and test data sets...\n")
# Data paths
DATA_TRAIN_PATH = "../data/train.csv"
DATA_TEST_PATH = "../data/test.csv"
# Load data
y_train, x_train, ids_train = load_csv_data(DATA_TRAIN_PATH)
y_test, x_test, ids_test = load_csv_data(DATA_TEST_PATH)


# Split the data in three groups in function of the "PRI num jet" value
jet_train_samples = get_jet_samples(x_train)
jet_test_samples = get_jet_samples(x_test)


# Define hyper-parameters for the hyper-tuned ridge regression
degrees = [12, 12, 12]
lambdas = [0.000147, 1e-3, 0.000464]



# Ridge regression and predictions
y_prediction_test = np.zeros(y_test.shape)
mean_accuracy = 0
mean_f1_score = 0

# Iterate through each of the three groups
for i in range(3):
    print("Computing ridge regression for group {ind}...".format(ind=i))
    
    # Parameters
    degree = degrees[i]
    lambda_ =lambdas[i]
    
    # Get train and test data
    train_index = jet_train_samples[i]
    test_index = jet_test_samples[i]
    
    x_tr, y_tr = x_train[train_index], y_train[train_index]
    x_te, y_te = x_test[test_index], y_test[test_index]
    
    # Clean train and test data
    x_tr,x_te = clean_data(x_tr, x_te)
    
    # Build polynomial data
    x_tr, y_tr = augment_data (x_tr, y_tr, degree)
    x_te, y_te = augment_data (x_te, y_te, degree)
    
    # Train model
    weights, loss = ridge_regression(y_tr, x_tr, lambda_)
    accuracy = predict_accuracy(y_tr, x_tr, weights)
    f1_score = compute_f1_score(y_tr, x_tr, weights)
    y_prediction_test[test_index] = predict_labels(weights, x_te)
    print("  Accuracy = {acc} \n  F1-score = {f1} \n".format(acc=accuracy, f1=f1_score))
    mean_accuracy += train_index.shape[0] * accuracy
    mean_f1_score += train_index.shape[0] * f1_score
    
mean_accuracy /= x_train.shape[0]
mean_f1_score /= x_train.shape[0]
print("Final accuracy = {acc} \nFinal F1-score = {f1} \n".format(acc=mean_accuracy, f1=mean_f1_score))



# Save ouput for submission
OUTPUT_PATH = "../data/submission.csv"
create_csv_submission(ids_test, y_prediction_test, OUTPUT_PATH)
print("File submission.csv ready to be submitted !")




