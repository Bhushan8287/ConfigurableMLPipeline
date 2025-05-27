from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR
from sklearn.neighbors import KNeighborsRegressor
from xgboost import XGBRegressor
from src.utils.logger_code import logger_for_model_training, log_component_start, log_component_end
from sklearn.metrics import r2_score, mean_squared_error, root_mean_squared_error, mean_absolute_error
from src.config.config_loader import get_config

def fit_evaluate_metrics(x_train, y_train, x_test, y_test):
    """
    Train multiple regression models and evaluate their performance on test data.

    This function fits a collection of regression models on the provided training data,
    evaluates them using RMSE, R2, MAE, and MSE on the test data, logs the metrics, 
    and returns the names of the top-performing models based on RMSE.

    Parameters
    ----------
    x_train : array-like or pd.DataFrame
        Feature matrix for training.
    
    y_train : array-like or pd.Series
        Target values for training.
    
    x_test : array-like or pd.DataFrame
        Feature matrix for testing.
    
    y_test : array-like or pd.Series
        Target values for testing.

    Returns
    -------
    list
        A list of model names corresponding to the top N models with the lowest RMSE, 
        where N is defined in the config file.
    """
    config = get_config()
    
    # Number of top models to return as defined in config file
    no_of_models_to_select = int(config['no_of_models_to_select_for_hyperparameter_tuning']['no_of_models'])

    try:
        # Log the start of the model training component
        log_component_start(logger_for_model_training, 'Model Training Component')
        logger_for_model_training.info('Model training started')

        # Define dictionary of regression models to train and evaluate
        models_dict = {
            "LinearRegression": LinearRegression(),
            "Ridge": Ridge(),
            "Lasso": Lasso(),
            "ElasticNet": ElasticNet(),
            "DecisionTreeRegressor": DecisionTreeRegressor(),
            "RandomForestRegressor": RandomForestRegressor(),
            "SVR": SVR(),
            "KNN": KNeighborsRegressor(),
            "XGboost": XGBRegressor()
        }

        model_metrics = []  # Store performance metrics for each model

        logger_for_model_training.info('Looping through models dictionary and training + evaluating')

        # Iterate through each model, train, predict, and record metrics
        for model_name, model in models_dict.items():
            model.fit(x_train, y_train)  # Train the model

            y_pred = model.predict(x_test)  # Predict on test data

            # Evaluate performance metrics
            r2 = r2_score(y_test, y_pred)
            rmse = root_mean_squared_error(y_test, y_pred)
            mae = mean_absolute_error(y_test, y_pred)
            mse = mean_squared_error(y_test, y_pred)

            # Append metrics to list
            metrics_dict = {
                'model_name': model_name,
                'rmse': rmse,
                'r2': r2,
                'mae': mae,
                'mse': mse,
            }
            model_metrics.append(metrics_dict)

            # Log metrics for current model
            logger_for_model_training.info(f'Model: {model_name} Metrics: RMSE: {rmse}, R2: {r2}, MAE: {mae}, MSE:{mse}')

        # Sort models by RMSE and select top N
        top_n_model_names = sorted(model_metrics, key=lambda x: x['rmse'])[:no_of_models_to_select]

        # Extract model names of top performers
        top_n_names = [model['model_name'] for model in top_n_model_names]

        logger_for_model_training.info('Model training has ended and metrics are returned')
        log_component_end(logger_for_model_training, 'Model Training Component')

        return top_n_names

    except Exception as mdltr_e:
        # Log any exceptions encountered
        logger_for_model_training.debug(f'Error encountered during model training: error {mdltr_e}')
        log_component_end(logger_for_model_training, 'Model Training Component')
        raise