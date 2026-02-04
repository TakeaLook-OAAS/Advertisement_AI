from loguru import logger

def setup_logger():
    # Customize as needed (file logging, rotation, etc.)
    logger.remove()
    logger.add(lambda msg: print(msg, end=""))
    return logger
