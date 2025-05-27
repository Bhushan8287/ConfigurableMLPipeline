from src.config.config_loader import get_config
from src.utils.logger_code import logger_for_cross_validation_component, log_component_start, log_component_end
from sklearn.linear_model import LinearRegression, Lasso, Ridge, ElasticNet
from sklearn.svm import SVR
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.model_selection import KFold
import numpy as np
from sklearn.model_selection import cross_val_score


def cross_validation_scores(xtrain_split, ytrain_split, return_metrics_dict):
    """
    Perform K-Fold cross-validation on multiple regression models, compute evaluation metrics and return a list of dictionaries containing the average
    cross-validation scores.

    Parameters
    ----------
    xtrain_split : array-like or pd.DataFrame
        The feature set used for training during cross-validation.

    ytrain_split : array-like or pd.Series
        The target values used for training during cross-validation.

    return_metrics_dict : str
        Indicates whether to return the cross-validation metrics dictionary.
        Accepts:
            - 'yes' : Returns a list of metric dictionaries for all models.
            - 'no'  : Returns None.

    Returns
    -------
    list[dict] or None
        A list of dictionaries, where each dictionary contains the model name and average cross-validation scores (R², RMSE, MAE, MSE) for that model.
        Returns None if `return_metrics_dict` is 'no'.

    Notes
    -----
    - The configuration file provides the number of folds (`n_splits`) and `random_state` for reproducibility.
    - The models include a range of regressors: linear, tree-based, SVM, KNN, and XGBoost.
    - The metrics are logged for traceability and debugging.
    """
    try:
        # Load configuration settings
        config = get_config()
        n_splits = config['K_fold']['n_splits']
        random_state = config['K_fold']['random_state_for_cv']

        log_component_start(logger_for_cross_validation_component, 'Cross Validation Component')
        logger_for_cross_validation_component.info('Cross validaton has started')

        # Define a dictionary of regression models to evaluate
        models_dict = {
        "LinearRegression": LinearRegression(),
        "Ridge": Ridge(),
        "Lasso": Lasso(),
        "ElasticNet": ElasticNet(),
        "DecisionTreeRegressor": DecisionTreeRegressor(),
        "RandomForestRegressor": RandomForestRegressor(),
        "SVR": SVR(),
        "XGBoostRegressor": XGBRegressor(),
        "KNN": KNeighborsRegressor()
        }
        
        # Set up K-Fold cross-validation with shuffle and random state from config
        cv_type = KFold(n_splits=n_splits, shuffle=True, random_state=random_state)
        
        # List to collect cross-validation metrics for each model
        cross_val_metrics_scores = []

        # Iterate through each model, perform cross-validation, and compute metrics
        for model_name, model in models_dict.items():
            # Compute mean R² score
            r2_mean = np.mean(cross_val_score(model, xtrain_split, ytrain_split, cv=cv_type, scoring='r2'))

            # Compute mean MAE (use negative scoring, hence negate result)
            mae_mean = -np.mean(cross_val_score(model, xtrain_split, ytrain_split, cv=cv_type, scoring='neg_mean_absolute_error'))

            # Compute mean RMSE
            rmse_mean = -np.mean(cross_val_score(model, xtrain_split, ytrain_split, cv=cv_type, scoring='neg_root_mean_squared_error'))

            # Compute mean MSE
            mse_mean = -np.mean(cross_val_score(model, xtrain_split, ytrain_split, cv=cv_type, scoring='neg_mean_squared_error'))

            # Log scores for current model
            logger_for_cross_validation_component.info(f'These are the cross validation scores for the model {model_name}')
            logger_for_cross_validation_component.info(f'This is average R2 score: {r2_mean:.4f}')
            logger_for_cross_validation_component.info(f'This is average RMSE score: {rmse_mean:.4f}')
            logger_for_cross_validation_component.info(f'This is average MAE score: {mae_mean:.4f}')
            logger_for_cross_validation_component.info(f'This is average MSE score: {mse_mean:.4f}')

            # Store metrics in a dictionary
            dict_metrics  = {
                'model_name': model_name,
                'rmse': rmse_mean,
                'r2': r2_mean,
                'mae': mae_mean,
                'mse': mse_mean,
            }

            # Append metrics to the results list
            cross_val_metrics_scores.append(dict_metrics)
        logger_for_cross_validation_component.info('Cross validation of all the models has completed')
        log_component_end(logger_for_cross_validation_component, 'Cross Validation Component')
        
        # Return the list of metrics if requested
        if return_metrics_dict == 'yes':
            logger_for_cross_validation_component.info('Returned a list of dictionaries consisting of cross validation scores')
            return cross_val_metrics_scores
        elif return_metrics_dict == 'no':
            return None

    except Exception as crsvl_e:
        # Log error and halt pipeline
        logger_for_cross_validation_component.debug(f'Error encountered during cross validation. error: {crsvl_e}')
        log_component_end(logger_for_cross_validation_component, 'Cross Validation Component')
        raise