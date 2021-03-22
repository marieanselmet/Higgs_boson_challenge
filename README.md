# Higgs Boson Machine Learning Challenge
Machine Learning - Project 1

**`Tshtsh_club`**: Marie Anselmet, Sofia Dandjee, Héloïse Monnet


## Run the project

1. Make sure that ```Python >= 3.7``` and ```NumPy >= 1.16``` are installed
2. Download the train and test data sets from [Kaggle competition dataset](https://www.kaggle.com/c/11051/download-all), and put ```train.csv``` and ```test.csv``` into a ```data\``` folder.
3. Go to `script\` folder and run ```run.py```. You will get ```submission.csv``` for Kaggle in the ```submission\``` folder.

~~~~shell
cd script
python run.py
~~~~

## Script files

### ```proj1_helpers.py```

- `sigmoid`: Sigmoid function.
- `load_csv_data`: Loads data from a csv file.
- `compute_f1_score`, `predict_accuracy`: Computes the accuracy and the F1 score of a prediction.
- `predict_labels`: Generates class predictions for a linear or a logistic regression. 
- `build_k_indices`, `cross_validation` : Generate the training and validation data for cross-validation.
- `classify`: Converts the (-1,1) of a label vector into (0,1), to use for the logistic regression.
- `batch_iter`: Generates a mini-batch for a dataset.
- `create_csv_submission`: Creates a csv output file for submission to Kaggle.

### ```cost.py```

- `compute_loss`: Computes the loss by mse for linear regression.
- `logistic_loss`: Compute the loss by negative log likelihood for the logistic regression.
- `reg_logistic_loss` : Compute the regularized logistic loss by negative log likelihood.

### ```compute_gradient.py```

- `compute_gradient`: Computes the gradient for the linear gradient descent.
- `logistic_gradient`: Compute the gradient for the logistic gradient descent.
- `reg_logistic_gradient`: Compute the gradient for the regularized logistic gradient descent.

### ```data_helpers.py```

- `get_jet_samples`: Divides the input data depending of their jet values.
- `clean_data`, `standardize` : Standardizes data, removes undefined values and features with a null standard deviation.
- `augment_data`, `build_model_data`,`build_poly_all_features`: Augment the data by building polynomial features.

### ```implementations.py```

- `least_squares_GD`: Linear regression using gradient descent.
- `least_squares_SGD`: Linear regression using stochastic gradient descent.
- `least_squares`: Least squares regression using normal equations.
- `ridge_regression`: Ridge regression using normal equations.
- `logistic_regression`: Logistic regression using stochastic gradient.
- `reg_logistic_regression`: Regularized logistic regression using stochastic gradient descent.

### ```run.py```

Script to produce the same .csv predictions used in the best submission on the Kaggle platform.



