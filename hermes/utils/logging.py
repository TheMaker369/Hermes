from loguru import logger
import sys
from hermes.utils.logging import logger

# Remove default Loguru handler
logger.remove()

# Add a custom handler for stderr
logger.add(sys.stderr,
           format="{time} {level} {message}",
           level="INFO")

# Add file logging (creates logs in a "logs" folder)
logger.add("logs/file_{time}.log",
           rotation="500 MB",
           retention="10 days",
           compression="zip")
