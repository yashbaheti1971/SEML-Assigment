"""Central logging configuration for the project.

All modules should import ``get_logger`` from this module and use it instead of
instantiating their own ``logging.Logger`` objects. This prevents duplicate
handlers and ensures logs are written to a rotating file in ``logs/app.log``.
"""

import logging
import os
from logging.handlers import RotatingFileHandler

# Ensure the ``logs`` directory exists adjacent to the project root.
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
LOG_DIR = os.path.join(BASE_DIR, "logs")
os.makedirs(LOG_DIR, exist_ok=True)

LOG_FILE = os.path.join(LOG_DIR, "app.log")

# Configure a rotating file handler: 5 MiB per file, keep up to 5 backups.
_file_handler = RotatingFileHandler(LOG_FILE, maxBytes=5 * 1024 * 1024, backupCount=5)
_file_handler.setFormatter(
    logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
)

_root_logger = logging.getLogger()
if not _root_logger.handlers:
    _root_logger.setLevel(logging.INFO)
    _root_logger.addHandler(_file_handler)

def get_logger(name: str) -> logging.Logger:
    """Return a logger that inherits the root configuration.

    ``name`` is typically ``__name__`` of the importing module.
    """
    return logging.getLogger(name)
