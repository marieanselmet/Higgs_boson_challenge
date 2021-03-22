import numpy as np

def get_jet_samples(tx):
    """ Return the indexes of the samples that have 0,1 and 2 respectively
        for the jet feature. """
    
    jet0_samples = np.where(tx[:,22]==0)[0] #indexes of samples with jet = 0
    jet1_samples = np.where(tx[:,22]==1)[0] #indexes of samples with jet = 1
    jet2_samples = np.where(tx[:,22]>=2)[0] #indexes of samples with jet = 2
    
    return [jet0_samples, jet1_samples, jet2_samples]
  
            
def standardize(x):
    """Standardize the data set x along the lines.
       Each feature has mean of 0 and std of 1
       after standardization. """
    
    mean = np.nanmean(x,axis=0)
    x = x - mean
    
    std = np.nanstd(x,axis=0)
    x = x / std
    
    return x, mean, std


def clean_data(tx_train, tx_test):
    """Remove undefined values and useless features and standardizes training and test data."""
    
    # Replace undefined values (equal to -999 in the data) by NaN 
    tx_train[tx_train == -999] = np.nan
    tx_test[tx_test == -999] = np.nan
    
    # Find columns full of NaN
    nan_features = list(np.where(np.all(np.isnan(tx_train), axis=0))[0])
    
    # Find features that have constant value (standard deviation equal to 0)
    constant_features = list(np.where(np.nanstd(tx_train, axis=0)==0)[0])
    
    # Remove selected columns
    indices = np.concatenate((nan_features, constant_features))
    
    tx_train = np.delete(tx_train, indices, axis=1)
    
    # Delete the same features in the test data
    tx_test = np.delete(tx_test, indices, axis=1)
    
    # Standardize
    tx_train, mean, std = standardize(tx_train)
    
    # Standardize test with train mean and std
    tx_test -= mean
    tx_test /= std
    
    # Standardize test with the training mean and std
    
    # Replace NaN values by 0    
    tx_train = np.nan_to_num(tx_train)
    tx_test = np.nan_to_num(tx_test)
    
    return tx_train, tx_test


def augment_data(tx, y, degree) :
    """Build polynomial features for an input data tx and add a column of 1
    for the bias term in the regression"""
    
    tx = build_poly_all_features (tx,degree)
    y,tx = build_model_data(tx,y)
    
    return tx, y


def build_model_data(x, y):
    """Form (y,tX) to get regression data in matrix form by adding a column of 1
    to tX for the bias term in the regression"""
    num_samples = len(y)
    tx = np.c_[np.ones(num_samples), x]
    return y, tx

def build_poly_all_features(x, degree):
    """ Augment all features of the input data x. """
    num_features = x.shape[1]
    tx = x
    
    for deg in range(2,degree+1):
        for feature in range(num_features):
            tx = np.c_[tx, x[:,feature]**deg]
    return tx



