import coloredlogs
import logging

logger = logging.getLogger("logger")
coloredlogs.install(level='DEBUG', logger=logger)
