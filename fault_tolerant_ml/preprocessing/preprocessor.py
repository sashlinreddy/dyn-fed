import numpy as np

def normalize_datasets(datasets):
    """
    This function computes mean and standard deviation of features in training
    dataset and normalizes both training and testing datasets accordingly. It
    also returns the corresponding mean and standard deviation vectors used 
    in normalization process.
    """
    training_mean = np.mean(datasets['training']['features'],axis=0)
    training_std = np.std(datasets['training']['features'],axis=0)
    
    datasets['training']['features'] = datasets['training']['features'] - training_mean
    datasets['training']['features'] = datasets['training']['features'] / training_std
    
    datasets['testing']['features'] = datasets['testing']['features'] - training_mean
    datasets['testing']['features'] = datasets['testing']['features'] / training_std
    
    norm_params = {'mean' : training_mean,
                  'std' : training_std}
    return norm_params