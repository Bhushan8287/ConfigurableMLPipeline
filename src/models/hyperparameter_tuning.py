from src.utils.logger_code import logger_for_hypertuning, log_component_start, log_component_end
from src.config.config_loader import get_config
from sklearn.metrics import root_mean_squared_error, mean_squared_error, mean_absolute_error, r2_score
from sklearn.model_selection import GridSearchCV
import joblib, json, os


def tune_models_and_log_metrics(models_with_paramgrid, xtrain, ytrain, xtest, ytest):
    """
    Perform hyperparameter tuning using GridSearchCV for a list of models, log metrics, and save the best models.

    This function iterates over a list of models with their respective hyperparameter grids, 
    performs GridSearchCV on each, evaluates them on the test set using standard regression metrics,
    logs the results, and saves the best parameters, metrics, and trained model to disk.

    Parameters
    ----------
    models_with_paramgrid : list of dict
        A list where each dictionary contains:
            - 'model': An instance of a scikit-learn estimator.
            - 'param_grid': A dictionary specifying the parameter grid to search over.
    xtrain : array-like
        Training features.
    ytrain : array-like
        Training labels.
    xtest : array-like
        Test features.
    ytest : array-like
        Test labels.

    Returns
    -------
    None
        The function saves metrics and models to disk and logs progress, but returns nothing.
    """
    # Start logging component
    log_component_start(logger_for_hypertuning, 'Hyperparameter_tuning_component')

    # Load configuration values
    config = get_config()
    save_metrics_and_models_path = config['paths']['outputs_directory']
    grid_cv = int(config['gridsearch_params']['cv'])
    grid_njobs = int(config['gridsearch_params']['n_jobs'])
    grid_scoring = config['gridsearch_params']['optimize_for']

    try:
        logger_for_hypertuning.info('Hyperparameter tuning started')

        # Loop through each model and its param grid
        for item in models_with_paramgrid:
            model = item['model']
            param_grid = item['param_grid']

            # Create a consistent model filename
            model_name = model.__class__.__name__
            filename_prefix = f"{model_name}_hypertuned"
            logger_for_hypertuning.info(f'Hypertuning model: {model_name}')

            # Set up and run GridSearchCV
            grid_search = GridSearchCV(
                estimator=model,
                param_grid=param_grid,
                cv=grid_cv,
                scoring=grid_scoring,
                n_jobs=grid_njobs,
                refit=True
            )

            grid_search.fit(xtrain, ytrain)

            # Get best model and hyperparameters
            best_model = grid_search.best_estimator_
            best_params = grid_search.best_params_

            logger_for_hypertuning.info(f'Hypertuning model: {model_name} completed')

            # Predict on test data
            predictions = best_model.predict(xtest)

            # Calculate evaluation metrics
            rmse = root_mean_squared_error(ytest, predictions)
            r2 = r2_score(ytest, predictions)
            mae = mean_absolute_error(ytest, predictions)
            mse = mean_squared_error(ytest, predictions)

            # Store metrics in a dictionary
            dict_store = {
                'metrics': {'rmse': rmse, 'r2': r2, 'mae': mae, 'mse': mse}
            }

            # Define paths for saving outputs
            params_path = os.path.join(save_metrics_and_models_path, f"{filename_prefix}_best_params.json")
            metrics_path = os.path.join(save_metrics_and_models_path, f"{filename_prefix}_metrics.json")
            joblib_path = os.path.join(save_metrics_and_models_path, f"{filename_prefix}.joblib")

            # Save best parameters
            with open(params_path, "w") as f:
                json.dump(best_params, f)

            # Save evaluation metrics
            with open(metrics_path, "w") as f:
                json.dump(dict_store, f)
            
            # Save the best model
            joblib.dump(best_model, joblib_path)

        logger_for_hypertuning.info('Hyperparameter tuning finished')
        log_component_end(logger_for_hypertuning, 'Hyperparameter_tuning_component')

    except Exception as hypr_e:
        # Log and re-raise any errors
        logger_for_hypertuning.debug(f'Error occured during hyperparameter tuning. error: {hypr_e}')
        log_component_end(logger_for_hypertuning, 'Hyperparameter_tuning_component')
        raise
