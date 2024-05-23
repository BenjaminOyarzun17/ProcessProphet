"""
this file contains a list of loggers that are used
for debugging purposes. 
"""
import logging

logger_generate_predictive_log= logging.getLogger('generate_predictive_log')
logger_generate_predictive_log.setLevel(logging.DEBUG)
fh_generate_predictive_log= logging.FileHandler('logs/generate_predictive_log.log')
fh_generate_predictive_log.setLevel(logging.DEBUG)
logger_generate_predictive_log.addHandler(fh_generate_predictive_log)




logger_multiple_prediction= logging.getLogger('multiple_prediction')
logger_multiple_prediction.setLevel(logging.DEBUG)
fh_multiple_prediction= logging.FileHandler('logs/multiple_prediction.log')
fh_multiple_prediction.setLevel(logging.DEBUG)
logger_multiple_prediction.addHandler(fh_multiple_prediction)


logger_single_prediction= logging.getLogger('single_prediction')
logger_single_prediction.setLevel(logging.DEBUG)
fh_single_prediction= logging.FileHandler('logs/single_prediction.log')
fh_single_prediction.setLevel(logging.DEBUG)
logger_single_prediction.addHandler(fh_single_prediction)



logger_import_event_log= logging.getLogger('import_event_log')
logger_import_event_log.setLevel(logging.DEBUG)
fh_import_event_log= logging.FileHandler('logs/import_event_log.log')
fh_import_event_log.setLevel(logging.DEBUG)
logger_import_event_log.addHandler(fh_import_event_log)



logger_get_dummy_process= logging.getLogger('get_dummy_process')
logger_get_dummy_process.setLevel(logging.DEBUG)
fh_get_dummy_process= logging.FileHandler('logs/get_dummy_process.log')
fh_get_dummy_process.setLevel(logging.DEBUG)
logger_get_dummy_process.addHandler(fh_get_dummy_process)



logger_split_train_test= logging.getLogger('split_train_test')
logger_split_train_test.setLevel(logging.DEBUG)
fh_split_train_test= logging.FileHandler('logs/split_train_test.log')
fh_split_train_test.setLevel(logging.DEBUG)
logger_split_train_test.addHandler(fh_split_train_test)


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

