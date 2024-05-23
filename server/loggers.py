"""
this file contains a list of loggers that are used
for debugging purposes. 
"""
import logging



logger_grid_search= logging.getLogger('grid_search')
logger_grid_search.setLevel(logging.DEBUG)
fh_grid_search= logging.FileHandler('logs/grid_search.log')
fh_grid_search.setLevel(logging.DEBUG)
logger_grid_search.addHandler(fh_grid_search)


logger_random_search= logging.getLogger('random_search')
logger_random_search.setLevel(logging.DEBUG)
fh_random_search= logging.FileHandler('logs/random_search.log')
fh_random_search.setLevel(logging.DEBUG)
logger_random_search.addHandler(fh_random_search)



logger_evaluate= logging.getLogger('evaluate')
logger_evaluate.setLevel(logging.DEBUG)
fh_evaluate= logging.FileHandler('logs/evaluate.log')
fh_evaluate.setLevel(logging.DEBUG)
logger_evaluate.addHandler(fh_evaluate)

