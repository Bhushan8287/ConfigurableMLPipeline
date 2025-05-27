import logging

# Logging config function
def get_logging_config():
    logging.basicConfig(level=logging.DEBUG, 
                    format='%(asctime)s-%(name)s-%(levelname)s-%(message)s',
                    datefmt='%Y-%m-%d %H:%M:%S',
                    handlers=[logging.FileHandler(r'C:\Users\BW\Desktop\ETL project\ML_project\logs\pipeline_logs.log'), # NOTE: update this path to change the location where the log file should be saved
                              logging.StreamHandler()]
                    )
    return logging


def log_component_start(logger, component_name: str):
    """
    Logs a standardized start message for a pipeline component.
    """
    logger.info(f"{'='*10} Starting {component_name} {'='*10}")

def log_component_end(logger, component_name: str):
    """
    Logs a standardized end message for a pipeline component, with spacing.
    """
    logger.info(f"{'='*10} Finished {component_name} {'='*10}\n")


logger_for_loading_data = get_logging_config().getLogger('Dataset_loading_component')
logger_for_loading_data.setLevel(get_logging_config().DEBUG)

logger_for_config_file = get_logging_config().getLogger('Loading_config_file_component')
logger_for_config_file.setLevel(get_logging_config().DEBUG)

logger_for_splitting_data = get_logging_config().getLogger('Data_splitting_component')
logger_for_splitting_data.setLevel(get_logging_config().DEBUG)

logger_for_cross_validation_component = get_logging_config().getLogger('Cross_validation_component')
logger_for_cross_validation_component.setLevel(get_logging_config().DEBUG)

logger_for_feature_selection_using = get_logging_config().getLogger('Feature_selection_component')
logger_for_feature_selection_using.setLevel(get_logging_config().DEBUG)

logger_for_feature_scaling = get_logging_config().getLogger('Feature_scaling_component')
logger_for_feature_scaling.setLevel(get_logging_config().DEBUG)

logger_for_model_training = get_logging_config().getLogger('Model_training_component')
logger_for_model_training.setLevel(get_logging_config().DEBUG)

logger_for_model_configurator = get_logging_config().getLogger('Model_configurator_component')
logger_for_model_configurator.setLevel(get_logging_config().DEBUG)

logger_for_hypertuning = get_logging_config().getLogger('Hyperparameter_tuning_component')
logger_for_hypertuning.setLevel(get_logging_config().DEBUG)

logger_for_pipeline_code = get_logging_config().getLogger('Pipeline_component')
logger_for_pipeline_code.setLevel(get_logging_config().DEBUG)


