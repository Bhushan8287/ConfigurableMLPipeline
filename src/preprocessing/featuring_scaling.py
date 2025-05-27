import pandas as pd
from src.utils.logger_code import logger_for_feature_scaling, log_component_start, log_component_end
from sklearn.preprocessing import StandardScaler, MinMaxScaler
import joblib
from src.config.config_loader import get_config
import os

def feature_scaling(scaling_method, xtrain, ytrain, xtest, ytest):    
    """
    Applies feature scaling to training and testing datasets using either StandardScaler or MinMaxScaler.
    
    Depending on the specified method, this function scales the input features (`xtrain` and `xtest`),
    logs the process, saves the fitted scaler object to disk for future use, and returns the scaled features
    along with the original target data.

    Parameters
    ----------
    scaling_method : str
        Method of scaling to apply. Must be either 'standard' for StandardScaler or 'minmax' for MinMaxScaler.
    xtrain : pandas.DataFrame
        Feature data for training.
    ytrain : pandas.Series or pandas.DataFrame
        Target data for training. Returned unchanged.
    xtest : pandas.DataFrame
        Feature data for testing.
    ytest : pandas.Series or pandas.DataFrame
        Target data for testing. Returned unchanged.

    Returns
    -------
    xtrain_scaled : pandas.DataFrame
        Scaled feature data for training with original column names preserved.
    ytrain : pandas.Series or pandas.DataFrame
        Unmodified target data for training.
    xtest_scaled : pandas.DataFrame
        Scaled feature data for testing with original column names preserved.
    ytest : pandas.Series or pandas.DataFrame
        Unmodified target data for testing.

    Raises
    ------
    Exception
        If an error occurs during the scaling process, the exception is logged and re-raised.
    """
    config = get_config()
    outputs = config['paths']['outputs_directory']

    if scaling_method == 'standard':
        try:
            # Log the beginning of the standard scaling process
            log_component_start(logger_for_feature_scaling, 'Feature Scaling Component')
            logger_for_feature_scaling.info('Scaling method selected is "Standard scaling" and scaling has started')

            # Apply StandardScaler to training and test feature data
            std_scaler = StandardScaler()
            xtrain_scaled = std_scaler.fit_transform(xtrain)
            xtest_scaled = std_scaler.transform(xtest)

            # Convert the scaled numpy arrays back into DataFrames with the original column names
            xtrain_std_scaled = pd.DataFrame(xtrain_scaled, columns=xtrain.columns)
            xtest_std_scaled = pd.DataFrame(xtest_scaled, columns=xtest.columns)

            # Save the trained scaler object to the outputs directory
            save_std_scaling_preprocessor = os.path.join(outputs, 'standard_scaler.joblib')
            joblib.dump(std_scaler, save_std_scaling_preprocessor)
            logger_for_feature_scaling.info('Standard scaler object saved')

            # Log completion and return the results
            logger_for_feature_scaling.info('X_train and X_test are standard scaled and the trained and test splits are returned')
            log_component_end(logger_for_feature_scaling, 'Feature Scaling Component')
            return xtrain_std_scaled, ytrain, xtest_std_scaled, ytest
        
        except Exception as std_e:
            # Log and re-raise any exceptions encountered during standard scaling
            logger_for_feature_scaling.info(f'Error encountered during standard scaling: {std_e}')
            log_component_end(logger_for_feature_scaling, 'Feature Scaling Component')
            raise

    elif scaling_method == "minmax":
        try:
            # Log the beginning of the min-max scaling process
            log_component_start(logger_for_feature_scaling, 'Feature Scaling Component')
            logger_for_feature_scaling.info('Scaling method selected is "Min-max scaling" and scaling has started')

            # Apply MinMaxScaler to training and test feature data
            minmax_scaler = MinMaxScaler()
            xtrain_scaled = minmax_scaler.fit_transform(xtrain)
            xtest_scaled = minmax_scaler.transform(xtest)

            # Convert the scaled numpy arrays back into DataFrames with the original column names
            xtrain_minmax_scaled = pd.DataFrame(xtrain_scaled, columns=xtrain.columns)
            xtest_minmax_scaled = pd.DataFrame(xtest_scaled, columns=xtest.columns)

            # Save the trained scaler object to the outputs directory
            save_mm_scaling_preprocessor = os.path.join(outputs, 'minmax_scaler.joblib')
            joblib.dump(minmax_scaler, save_mm_scaling_preprocessor)
            logger_for_feature_scaling.info('Min-max scaler object saved')

            # Log completion and return the results
            logger_for_feature_scaling.info('X_train and X_test are min-max scaled and the trained and test splits are returned')
            log_component_end(logger_for_feature_scaling, 'Feature Scaling Component')
            return xtrain_minmax_scaled, ytrain, xtest_minmax_scaled, ytest

        except Exception as mm_e:
            # Log and re-raise any exceptions encountered during min-max scaling
            logger_for_feature_scaling.info(f'Error encountered during min-max scaling: {mm_e}')
            log_component_end(logger_for_feature_scaling, 'Feature Scaling Component')
            raise
