from src.utils.logger_code import logger_for_model_configurator, log_component_start, log_component_end
from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet
from sklearn.neighbors import KNeighborsRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR
from sklearn.tree import DecisionTreeRegressor
from xgboost import XGBRegressor

# Instantiate regression models
rndf_model = RandomForestRegressor()
svr_model = SVR()
ridge_model = Ridge()
lasso_model = Lasso()
ela_net_model = ElasticNet()
dtree_model = DecisionTreeRegressor()
xgbst_model = XGBRegressor()
lr_model = LinearRegression()
knn_model = KNeighborsRegressor()

# Define hyperparameter grid for Decision Tree Regressor
param_grid_dtree = {
    "max_depth": [3, 5, 10, 15, 20, None],
    "min_samples_split": [2, 5, 10, 20],
    "min_samples_leaf": [1, 2, 4, 8, 10],
    "max_features": ["sqrt", "log2", None],
    "ccp_alpha": [0.0, 0.01, 0.02, 0.05] 
}

# Define hyperparameter grid for XGBoost Regressor
param_grid_xgb = {
    "n_estimators": [50, 100, 200],
    "learning_rate": [0.01, 0.05, 0.1, 0.2],
    "max_depth": [3, 5, 10, 15],
    "min_child_weight": [1, 3, 5],
    "subsample": [0.6, 0.8, 1.0],
    "colsample_bytree": [0.6, 0.8, 1.0],
    "gamma": [0, 0.1, 0.2, 0.5]
}

# Define hyperparameter grid for Random Forest Regressor
param_grid_rf = {
    "n_estimators": [50, 100, 200],
    "max_depth": [3, 5, 10, 15, None],
    "min_samples_split": [2, 5, 10],
    "min_samples_leaf": [1, 2, 4],
    "max_features": ["sqrt", "log2", None],
    "bootstrap": [True, False]
}

# Define hyperparameter grid for KNN Regressor
param_grid_knn = {
    "n_neighbors": [3, 5, 7, 10],
    "weights": ["uniform", "distance"],
    "metric": ["euclidean", "manhattan", "minkowski"],
    "p": [1, 2]
}

# Define hyperparameter grid for Linear Regression
param_grid_lr = {
    'fit_intercept': [True, False],
    'normalize': [False]  # Deprecated; retained for backward compatibility
}

# Define hyperparameter grid for Ridge Regression
param_grid_ridge = {
    'alpha': [0.01, 0.1, 1.0, 10.0, 100.0],
    'fit_intercept': [True, False],
    'solver': ['auto', 'svd', 'cholesky', 'lsqr', 'saga']
}

# Define hyperparameter grid for Lasso Regression
param_grid_lasso = {
    'alpha': [0.0001, 0.001, 0.01, 0.1, 1.0],
    'fit_intercept': [True, False],
    'selection': ['cyclic', 'random']
}

# Define hyperparameter grid for ElasticNet Regression
param_grid_elastic = {
    'alpha': [0.0001, 0.001, 0.01, 0.1, 1.0],
    'l1_ratio': [0.1, 0.3, 0.5, 0.7, 0.9],
    'fit_intercept': [True, False],
    'selection': ['cyclic', 'random']
}

# Define hyperparameter grid for Support Vector Regressor
param_grid_svr = {
    'kernel': ['linear', 'rbf'],
    'C': [0.1, 1.0, 10.0, 100.0],
    'epsilon': [0.01, 0.1, 0.5, 1.0],
    'gamma': ['scale', 'auto']
}

def get_selected_model_grids(selected_models_names_list):
    """
    Returns a list of selected regression models and their hyperparameter grids.

    This function maps each model name in the input list to its corresponding 
    model object and hyperparameter grid. It is used to dynamically fetch only 
    the models and their hyper-paramters selected by the model training component.

    Parameters
    ----------
    selected_models_names_list : list of str
        List of model names to be selected from available regressors.
        Valid names include:
        ['LinearRegression', 'KNN', 'RandomForestRegressor', 'SVR', 
         'Ridge', 'Lasso', 'ElasticNet', 'DecisionTreeRegressor', 'XGboost']

    Returns
    -------
    models_with_paramgrid : list of dict
        A list where each element is a dictionary with:
        - 'model': instantiated sklearn/model
        - 'param_grid': dictionary of hyperparameters for that model

    Raises
    ------
    Exception
        Any exception that occurs during the configuration process is logged 
        and re-raised for higher-level handling.
    """
    try:
        log_component_start(logger_for_model_configurator, 'Model Configurator Component')
        
        # Map of model names to their objects and parameter grids
        model_param_map = {
            'LinearRegression': (lr_model, param_grid_lr),
            'KNN': (knn_model, param_grid_knn),
            'RandomForestRegressor': (rndf_model, param_grid_rf),
            'SVR': (svr_model, param_grid_svr),
            'Ridge': (ridge_model, param_grid_ridge),
            'Lasso': (lasso_model, param_grid_lasso),
            'ElasticNet': (ela_net_model, param_grid_elastic),
            'DecisionTreeRegressor': (dtree_model, param_grid_dtree),
            'XGboost': (xgbst_model, param_grid_xgb)
        }

        models_with_paramgrid = []

        # Filter and append only selected models
        for model_name in selected_models_names_list:
            if model_name in model_param_map:
                model, param_grid = model_param_map[model_name]
                models_with_paramgrid.append({
                    'model': model,
                    'param_grid': param_grid
                })

        logger_for_model_configurator.info('Returned the list of dictionaries with models and their corresponding param grids')
        log_component_end(logger_for_model_configurator, 'Model Configurator Component')
        return models_with_paramgrid

    except Exception as mdl_configr:
        logger_for_model_configurator.debug(f'Error encountered during model configuration. error: {mdl_configr}')
        log_component_end(logger_for_model_configurator, 'Model Configurator Component')
        raise
