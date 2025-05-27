from sklearn.model_selection import train_test_split
from src.config.config_loader import get_config
from src.utils.logger_code import logger_for_splitting_data, log_component_start, log_component_end


def split_dataset(loaded_datadf):
    """
    Splits the input dataset into training and testing sets for features (X) and target (y).

    The function performs the following:
        1. Loads the test size and random state from `config.json` via `get_config()`.
        2. Splits the input DataFrame into:
           - Features (X): All columns except 'Daily_Revenue'.
           - Target (y): The 'Daily_Revenue' column.
        3. Performs a train-test split using scikit-learn's `train_test_split`.

    Logging tracks each stage of the splitting process including the shapes of the resulting splits.

    Args:
        loaded_datadf (pd.DataFrame): The raw dataset loaded from CSV.

    Returns:
        tuple: (X_train, X_test, y_train, y_test) — all as pandas DataFrames/Series.

    Raises:
        Exception: If the split fails, logs the error and re-raises the exception.
    """
    try:
        log_component_start(logger_for_splitting_data, 'Splitting Data Component')
        # Load config values for splitting
        config = get_config()
        test_size = config['splitting']['test_size']
        random_state = int(config['splitting']['random_state'])

        logger_for_splitting_data.info('The dataset is now being split into "X", "y" and "train" and "test" splits')

        # Separate features and target
        X = loaded_datadf.drop(columns='Daily_Revenue')
        y = loaded_datadf['Daily_Revenue']

        # Perform train-test split
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state)

        logger_for_splitting_data.info(f'Split shapes X_train: {X_train.shape}, X_test: {X_test.shape}, 'f'y_train: {y_train.shape}, y_test: {y_test.shape}')
        logger_for_splitting_data.info('The dataset has been successfully split into training and testing sets and are returned')

        log_component_end(logger_for_splitting_data, 'Splitting Data Component')
        return X_train, y_train, X_test, y_test
    
    # if error is encountered it will be logged and the pipeline flow will halt(stop)
    except Exception as split_e:
        logger_for_splitting_data.debug(f'The was an error when tried to split the dataset. error {split_e}')
        log_component_end(logger_for_splitting_data, 'Splitting Data Component')
        raise