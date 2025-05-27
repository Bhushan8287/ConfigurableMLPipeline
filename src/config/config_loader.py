import json
from src.utils.logger_code import logger_for_config_file, log_component_start, log_component_end

def load_config_file():
    """
    Loads the project's configuration from a JSON file located at a hard-coded path.

    This function opens and reads the `config.json` file from a specified location, parses its contents using the `json` module, and returns the resulting dictionary.
    
    It also logs success and failure messages for debugging and monitoring.

    Returns:
        dict: The contents of the configuration file as a Python dictionary.

    Raises:
        Exception: If the file cannot be loaded or parsed, an exception is raised and logged.
    """
    try:
    # Log the attempt to load the config file
        log_component_start(logger_for_config_file, 'Config File Component')
        logger_for_config_file.info('Trying to load the config file to access the data inside it')

        # NOTE: Update this hard-coded path if the project is moved to another directory
        config_file_path = r'C:\Users\BW\Desktop\ETL project\ML_project\config.json'
        
        # Open and load the contents of the config file using JSON
        with open(config_file_path, 'r') as config_file:
            config_file_loaded = json.load(config_file)
            logger_for_config_file.info('Config file has been succesfully loaded')
            log_component_end(logger_for_config_file, 'Config File Component')

            # Return the parsed configuration data
            return config_file_loaded
    except Exception as cfg_e:
        # Log any exception encountered and re-raise it to halt execution
        logger_for_config_file.debug(f'Error loading config file. error: {cfg_e}')
        log_component_end(logger_for_config_file, 'Config File Component')
        raise

# Global configuration dictionary loaded once for reuse across the codebase
config = load_config_file()

def get_config():
    """
    Provides access to the loaded configuration dictionary.

    This is a getter function that returns the global config variable loaded
    during module initialization.

    Returns:
        dict: The configuration settings.
    """
    return config