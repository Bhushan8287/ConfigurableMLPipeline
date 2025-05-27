from src.data.load_data import load_data
from src.preprocessing.splitting_data import split_dataset
from src.preprocessing.feature_selection import feature_selection_methods
from src.preprocessing.cross_validation import cross_validation_scores
from src.preprocessing.featuring_scaling import feature_scaling
from src.models.model_training import fit_evaluate_metrics
from src.models.model_configurator import get_selected_model_grids
from src.models.hyperparameter_tuning import tune_models_and_log_metrics
from src.utils.logger_code import logger_for_pipeline_code
from src.config.config_loader import get_config


def Pipeline():
    """
    Executes the full machine learning pipeline from data loading to hyperparameter tuning.

    This pipeline includes the following stages:
    - Load dataset
    - Split dataset into train/test
    - Evaluate baseline cross-validation scores
    - Perform feature selection
    - Re-evaluate cross-validation scores
    - Apply feature scaling
    - Train models and evaluate their metrics
    - Select top-performing models and perform hyperparameter tuning

    Logging is performed at each step to track progress and results.

    Returns
    -------
    None
        All metrics, models, and logs are stored in specified output directories.
    """
    config = get_config()
    feature_selection_method = config['feature_selection']['selected_feature_selection_method']
    feature_scaling_method = config['scaling_method']['scaler']
    try:
        logger_for_pipeline_code.info('Pipeline execution has started')

        # Step 1: Load the dataset
        dataset_loaded = load_data()

        # Step 2: Split dataset into training and test sets
        X_train, y_train, X_test, y_test = split_dataset(loaded_datadf=dataset_loaded)

        # Step 3: Baseline model performance using cross-validation (pre-feature selection)
        cross_validation_scores(xtrain_split=X_train, ytrain_split=y_train, return_metrics_dict='no')

        # Step 4: Feature selection based on configured method
        xtrain_selected_features, y_train, xtest_selected_features, y_test = feature_selection_methods(
            method=feature_selection_method, 
            xtrain=X_train, 
            ytrain=y_train, 
            xtest=X_test, 
            ytest=y_test
        )

        # Step 5: Re-evaluate model using cross-validation (post-feature selection)
        cross_validation_scores(xtrain_split=xtrain_selected_features, ytrain_split=y_train, return_metrics_dict='no')

        # Step 6: Apply feature scaling on selected features
        xtrain_scaled, y_train, xtest_scaled, y_test = feature_scaling(
            scaling_method=feature_scaling_method, 
            xtrain=xtrain_selected_features, 
            ytrain=y_train, 
            xtest=xtest_selected_features, 
            ytest=y_test
        )

        # Step 7: Train models and evaluate them on test data
        top_performing_model_names = fit_evaluate_metrics(
            x_train=xtrain_scaled, 
            y_train=y_train, 
            x_test=xtest_scaled, 
            y_test=y_test
        )

        # Step 8: Retrieve model objects and their respective hyperparameter grids
        selected_models_and_param_grids = get_selected_model_grids(top_performing_model_names)

        # Step 9: Hyperparameter tuning and final evaluation + model saving
        tune_models_and_log_metrics(
            models_with_paramgrid=selected_models_and_param_grids, 
            xtrain=xtrain_scaled, 
            ytrain=y_train, 
            xtest=xtest_scaled, 
            ytest=y_test
        )
        logger_for_pipeline_code.info('Pipeline execution has finished')
    except Exception as pl_e:
        # Log if any step in the pipeline fails
        logger_for_pipeline_code.debug(f'Error encountered during execution of the pipeline. error {pl_e}')
