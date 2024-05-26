from flask import Blueprint, request, send_file, make_response, jsonify
from preprocessing import * 
from nn_manager import *
from process_model_manager  import *
from prediction_manager import *

routes =Blueprint("Routes", __name__)


ok = {"status":"OK"}

@routes.route('/start')
def start():
    return ok



@routes.route('/single_prediction', methods = ["GET"])
def single_prediction():
    """
    required parameters: 
    - model
    - config? 
    -  
    
    """



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
    :para  cuda: use cuda.
    """
    if request.method == 'GET':
        request_config = request.args.to_dict()
        is_xes = True if request_config["is_xes"]=="True" else False
        cuda = True if request_config["cuda"]=="True" else False
        path_to_log = str(request_config["path_to_log"])
        path_to_log = "/home/benja/Desktop/SPP-process-discovery/data/train_day_joined.csv"       
        case_id= str(request_config["case_id"])
        activity= str(request_config["activity_key"])
        timestamp= str(request_config["timestamp_key"])

        config = Config()
        #config = config.load_config(request_config["config"])

        preprocessor = Preprocessing()
        preprocessor.handle_import(is_xes, path_to_log, case_id, timestamp, activity)
        train, test= preprocessor.split_train_test(float(request_config["split"]))

        nn_manager= NNManagement(None) 
        nn_manager.config.cuda = cuda
        nn_manager.config.absolute_frequency_distribution = preprocessor.absolute_frequency_distribution
        nn_manager.config.number_classes = preprocessor.number_classes
        nn_manager.config.case_id_le = preprocessor.case_id_le
        nn_manager.config.activity_le = preprocessor.activity_le

        nn_manager.load_data(train, test, preprocessor.case_id_key, preprocessor.case_timestamp_key, preprocessor.case_activity_key)
        nn_manager.train()

        training_stats = nn_manager.get_training_statistics()
        config = nn_manager.config.asdict()

        nn_manager.export_nn_model()

        model_path = "model.pt"
        with open(model_path, 'rb') as f:
            model_data = f.read()
        response = make_response(model_data)

        metadata = json.dumps({
            "training_statistics": training_stats, 
            "config": config
        })

        # TODO: it might be convenient to also send the nn_mnager config.
        response.headers.set('Content-Type', 'application/octet-stream') # announce file included
        response.headers.set('Content-Disposition', 'attachment', filename='model.pt') 
        response.headers.set('X-Metadata', metadata) #include json metadata

        return response 






@routes.route('/end_session')
def end_session():

    return ok

