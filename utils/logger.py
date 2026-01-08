import logging
import os

def setup_logger(log_file):
    log_dir = os.path.dirname(log_file)
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
        print(f"The log directory '{log_dir}' does not exist and has been automatically created.")

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler(log_file, mode='a', encoding='utf-8')
        ]
    )
    logger = logging.getLogger(__name__)
    return logger

logger = setup_logger(log_file="./log/run.log")