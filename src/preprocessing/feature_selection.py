import pandas as pd
from src.utils.logger_code import logger_for_feature_selection_using, log_component_start, log_component_end
from src.config.config_loader import get_config
from sklearn.ensemble import RandomForestRegressor
from sklearn.feature_selection import SelectKBest, mutual_info_regression

# Custom exception raised when no features meet the selection criteria for pcc
class NoFeaturesSelectedError(Exception):
    """Exception raised when no features are selected based on the given criteria when using pcc."""
    pass


def feature_selection_methods(method, xtrain, ytrain, xtest, ytest):
    """
    Perform feature selection on training and test datasets using the specified method.

    Parameters
    ----------
    method : str
        The method to use for feature selection. Supported values are:
        'corr' for Pearson Correlation Coefficient (PCC),
        'feature_importance' for Random Forest feature importances,
        'mutual_info' for mutual information regression with SelectKBest.

    xtrain : pandas.DataFrame
        Training feature set.

    ytrain : pandas.Series or pandas.DataFrame
        Target variable for training.

    xtest : pandas.DataFrame
        Test feature set.

    ytest : pandas.Series or pandas.DataFrame
        Target variable for test.

    Returns
    -------
    Tuple[pandas.DataFrame, pandas.Series, pandas.DataFrame, pandas.Series]
        Tuple containing:
        - Reduced training features
        - Corresponding training targets
        - Reduced test features
        - Corresponding test targets

    Raises
    ------
    NoFeaturesSelectedError
        If no features meet the selection criteria when using the 'corr' method.

    Exception
        Any exception raised during model fitting or feature selection process is logged and re-raised.
    """
    config = get_config()  # Load configuration settings

    if method == 'corr':
        try:
            log_component_start(logger_for_feature_selection_using, 'Feature Selection Component')
            logger_for_feature_selection_using.info('Feature selection process has started using "PCC" method.')

            # Combine xtrain and ytrain into a single DataFrame to compute correlation
            train_dataset = xtrain.copy()
            train_dataset['Daily_Revenue'] = ytrain

            # Compute Pearson correlation matrix
            correlation_matrix = train_dataset.corr()

            # Get correlations with the target variable
            target_correlations = correlation_matrix['Daily_Revenue'].sort_values(ascending=False)
            target_correlations.drop(index='Daily_Revenue', inplace=True)

            selected_features = []
            selected_features_correlation_values = []

            # Use configured threshold to filter relevant features
            threshold = config['feature_selection']['corr_threshold']

            for feature_name, corr_value in target_correlations.items():
                if corr_value > threshold:
                    selected_features.append(feature_name)
                    selected_features_correlation_values.append(corr_value)

            logger_for_feature_selection_using.info(f'Selected features: {selected_features} with correlation values: {selected_features_correlation_values}')

            if not selected_features:
                message = ('No features meet the correlation threshold. ' 'Consider adjusting the threshold in the configuration file.')
                logger_for_feature_selection_using.info(message)
                raise NoFeaturesSelectedError(message)

            # Subset train and test datasets using the selected features
            xtrain_selected_features_pcc = xtrain[selected_features]
            xtest_selected_features_pcc = xtest[selected_features]

            logger_for_feature_selection_using.info('Feature selection process has completed using "PCC" method and train and test splits are returned')
            log_component_end(logger_for_feature_selection_using, 'Feature Selection Component')
            return xtrain_selected_features_pcc, ytrain, xtest_selected_features_pcc, ytest

        except Exception as fe_sec_e:
            logger_for_feature_selection_using.debug(f'Error during feature selection using PCC: {fe_sec_e}')
            log_component_end(logger_for_feature_selection_using, 'Feature Selection Component')
            raise

    elif method == 'feature_importance':
        try:
            log_component_start(logger_for_feature_selection_using, 'Feature Selection Component')
            logger_for_feature_selection_using.info('Feature selection process has started using "Feature importance with random forest" method')

            # Train a Random Forest model
            rf = RandomForestRegressor(random_state=42)
            rf.fit(xtrain, ytrain)

            # Get feature importances from trained model
            importances = pd.Series(data=rf.feature_importances_, index=xtrain.columns)

            no_of_features = int(config['feature_selection']['no_of_features_to_select_using_feature_imp'])

            # Select top N features based on importance
            top_features = importances.sort_values(ascending=False).head(no_of_features)

            xtrain_selected_features_feat_imp = xtrain[top_features.index]
            xtest_selected_features_feat_imp = xtest[top_features.index]

            logger_for_feature_selection_using.info(f'Selected features: {xtrain_selected_features_feat_imp.columns.tolist()} with their feature importances values: {top_features.to_list()}')

            logger_for_feature_selection_using.info('Feature selection process has completed using "Feature importance with random forest" method and train and test splits are returned')
            log_component_end(logger_for_feature_selection_using, 'Feature Selection Component')
            return xtrain_selected_features_feat_imp, ytrain, xtest_selected_features_feat_imp, ytest

        except Exception as fe_e:
            logger_for_feature_selection_using.debug(f'Error during feature selection using method Feature importances: {fe_e}')
            log_component_end(logger_for_feature_selection_using, 'Feature Selection Component')
            raise

    elif method == 'mutual_info':
        try:
            log_component_start(logger_for_feature_selection_using, 'Feature Selection Component')
            logger_for_feature_selection_using.info('Feature selection process has started using "Mutual info regression method with select k best" method')

            k_features = int(config['feature_selection']['features_to_select_k_using_mutual_info'])

            # Apply SelectKBest with mutual information scoring
            selector = SelectKBest(score_func=mutual_info_regression, k=k_features)
            selector.fit(xtrain, ytrain)

            mi_scores = pd.Series(data=selector.scores_, index=xtrain.columns)

            sorted_scores = mi_scores.sort_values(ascending=False)
            top_features = sorted_scores.head(k_features).index.tolist()

            xtrain_selected_features_mutualinfo = xtrain[top_features]
            xtest_selected_features_mutualinfo = xtest[top_features]

            logger_for_feature_selection_using.info(f'Selected features: {xtrain_selected_features_mutualinfo.columns.tolist()} with their mutual info values: {sorted_scores[:k_features].tolist()}')

            logger_for_feature_selection_using.info('Feature selection process has completed using "Mutual info regression method with select k best" method and train and test splits are returned')

            log_component_end(logger_for_feature_selection_using, 'Feature Selection Component')
            return xtrain_selected_features_mutualinfo, ytrain, xtest_selected_features_mutualinfo, ytest
        

        except Exception as mi_e:
            logger_for_feature_selection_using.debug(f'Error during feature selection using method mutual info: {mi_e}')
            log_component_end(logger_for_feature_selection_using, 'Feature Selection Component')
            raise
