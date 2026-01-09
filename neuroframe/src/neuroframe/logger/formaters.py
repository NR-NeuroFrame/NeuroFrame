# ================================================================
# 0. Section: Imports
# ================================================================
import logging
from .filters import FunctionSeparatorFilter



# ================================================================
# 1. Section: Formatters
# ================================================================
detail_format = logging.Formatter(
    "{separator}{asctime}.{msecs:03.0f}: {levelname} [{module}.{funcName}] - {message}",
    style="{",
    datefmt="%Y-%m-%d %H:%M:%S",
)


console_format = logging.Formatter(
    "{levelname} - {message}",
    style="{",
)