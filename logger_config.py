import os
import logging
from pydantic_loggings.base import Logging

def setup_app_logging():
    # 1. Get the base logger from your library
    logger = Logging().get_logger(configure=True)

    # 2. Create directory
    if not os.path.exists("logs"):
        os.makedirs("logs")

    # 3. Setup File Handler
    file_handler = logging.FileHandler("logs/app.log")
    file_handler.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    file_handler.setFormatter(formatter)

    # 4. Add handler and return
    logger.addHandler(file_handler)
    return logger