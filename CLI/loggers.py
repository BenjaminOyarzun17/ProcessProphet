
"""
this file contains a list of loggers that are used
for debugging purposes. 
"""
import logging

logger_set_params_cli= logging.getLogger('set_params_cli')
logger_set_params_cli.setLevel(logging.DEBUG)
fh_set_params_cli= logging.FileHandler('logs/set_params_cli.log')
fh_set_params_cli.setLevel(logging.DEBUG)
logger_set_params_cli.addHandler(fh_set_params_cli)



