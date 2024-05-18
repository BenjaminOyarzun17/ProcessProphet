from flask import Blueprint, request
from state import *
from preprocessing import * 
from nn_manager import *

routes =Blueprint("Routes", __name__)


ok = {"status":"OK"}

@routes.route('/start')
def start():
    return ok


@routes.route('/start_session')
def start_session():
    return ok





@routes.route('/train_nn', methods = ["GET"])
def train_nn():
    """
    :param path_to_log: the program will only search the path ../projects/<subfolder>/<file_name> 
    so the extected addr is of the form `/subfolder/file_name`.
    :param is_xes: Boolean expected. if False, then csv expected.
    :param training_params: dictionary of params for the nn expected. 
    See Config class for possible values.  
    :param path: path to the event log. just used if is_xes is False
    :param case_id: case id column name. just used if is_xes is False
    :param activity_key: activity column name. just used if is_xes is False
    """
    if request.method == 'GET':
        request_config = request.args.to_dict()
        is_xes = True if request_config["is_xes"]=="True" else False
        path_to_log = str(request_config["path_to_log"])
        path_to_log = "/home/benja/Desktop/SPP-process-discovery/data/train_day_joined.csv"       
        preprocessor = Preprocessing()
        nn_manager = NNManagement()
        #preprocessor.check_path(path_to_log)
        print(request_config)
        if is_xes:
            preprocessor.import_event_log_xes(path_to_log)
        else: 
            preprocessor.import_event_log_csv(
                path_to_log,
                request_config["case_id"], 
                request_config["activity_key"] ,
                request_config["timestamp_key"],
                request_config["sep"])

        train, test, no_classes = preprocessor.split_train_test(.9)
        nn_manager.train(
            train,
            test,
            preprocessor.case_id_key,
            preprocessor.case_timestamp_key,
            preprocessor.case_activity_key,
            no_classes
        )
        stats_in_json = nn_manager.get_training_statistics()
        return stats_in_json
         







@routes.route('/end_session')
def end_session():

    return ok

