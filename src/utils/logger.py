import logging
import os
from datetime import datetime

def setup_logger(name="process", log_dir="logs"):
    """
    Sets up a standard Python logger that writes output to the console 
    and to a timestamped file to preserve proof of execution.

    Args:
        name (str): Identifier for the logger.
        log_dir (str): Directory where the .log file will be saved.

    Returns:
        logging.Logger: The configured logger instance.
    """
    os.makedirs(log_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = os.path.join(log_dir, f"{name}_{timestamp}.log")
    
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)
    
    # Avoid duplicating handlers if logger is repeatedly instantiated
    if not logger.handlers:
        formatter = logging.Formatter('%(asctime)s | %(name)s | [%(levelname)s] | %(message)s', datefmt='%Y-%m-%d %H:%M:%S')
        
        # File Handler: writes to the permanent log file
        fh = logging.FileHandler(log_file)
        fh.setFormatter(formatter)
        logger.addHandler(fh)
        
        # Console Handler: outputs to terminal/notebook
        ch = logging.StreamHandler()
        ch.setFormatter(formatter)
        logger.addHandler(ch)
        
    return logger
