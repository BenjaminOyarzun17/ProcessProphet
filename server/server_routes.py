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





@routes.route('/generate_predictive_log', methods = ["GET"])
def generate_predictive_log():
    
    if request.method == 'GET':
        request_config = request.args.to_dict()
        is_xes = True if request_config["is_xes"]=="True" else False
        

        case_id= str(request_config["case_id"])
        activity= str(request_config["activity_key"])
        timestamp= str(request_config["timestamp_key"])
        path_to_log = str(request_config["path_to_log"])
        path_to_model = str(request_config["path_to_model"])
        cuda = True if request_config["cuda"]=="True" else False
        non_stop = bool(request_config["non_stop"])
        upper = int(request_config["upper"])
        random_cuts =bool(request_config["random_cuts"])
        cut_length = int(request_config["cut_length"])
        dic =  json.loads(request_config.get("config"))
        new_log_path=  request_config.get("new_log_path")


        preprocessor = Preprocessing()
        if is_xes:
            #preprocessor.import_event_log_xes(path , "case:concept:name", "concept:name", "time:timestamp")# hospital
            preprocessor.import_event_log_xes(path_to_log , case_id, activity, timestamp)# bpi 2019
        else:
            preprocessor.import_event_log_csv(path_to_log , case_id, activity, timestamp, ',')


        config = Config()
        config.load_config(dic)

        nn_manager = NNManagement(config)
        nn_manager.import_nn_model(path_to_model)

        pmm = ProcessModelManager(
            preprocessor.event_df, 
            nn_manager.model, 
            nn_manager.config,
            preprocessor.case_activity_key,
            preprocessor.case_id_key,
            preprocessor.case_timestamp_key
        )
        pmm.end_activities = preprocessor.find_end_activities()



        #: TODO: check the combinations for the bool attributes, some combinations 
        # are not well defined.
        pmm.generate_predictive_log(non_stop=non_stop, upper =upper, random_cuts=random_cuts,cut_length=cut_length, new_log_path = new_log_path )

        
        print("event log generated")
        
        
        return ok #they are already encoded











@routes.route('/multiple_prediction', methods = ["GET"])
def multiple_prediction():
    if request.method == 'GET':
        request_config = request.args.to_dict()
        is_xes = True if request_config["is_xes"]=="True" else False

        depth = int(request_config["depth"])
        degree = int(request_config["degree"])
        case_id= str(request_config["case_id"])
        activity= str(request_config["activity_key"])
        timestamp= str(request_config["timestamp_key"])
        path_to_log = str(request_config["path_to_log"])
        preprocessor = Preprocessing()

        if is_xes:
            #preprocessor.import_event_log_xes(path , "case:concept:name", "concept:name", "time:timestamp")# hospital
            preprocessor.import_event_log_xes(path_to_log , case_id, activity, timestamp)# bpi 2019
        else:
            preprocessor.import_event_log_csv(path_to_log , case_id, activity, timestamp, ',')

        input_df = preprocessor.event_df

        cuda = True if request_config["cuda"]=="True" else False

        path_to_model = str(request_config["path_to_model"])

        config = Config()
        dic =  json.loads(request_config.get("config"))
        config.load_config(dic)

        nn_manager = NNManagement(config)
        nn_manager.import_nn_model(path_to_model)
        pm = PredictionManager(
            nn_manager.model, 
            case_id, 
            activity,
            timestamp, 
            config
        )

        pm.multiple_prediction_dataframe(
            depth, 
            degree, 
            input_df
        )
        paths = pm.jsonify_paths()
        
        return paths #they are already encoded


@routes.route('/single_prediction', methods = ["GET"])
def single_prediction():
    if request.method == 'GET':
        request_config = request.args.to_dict()
        is_xes = True if request_config["is_xes"]=="True" else False

        case_id= str(request_config["case_id"])
        activity= str(request_config["activity_key"])
        timestamp= str(request_config["timestamp_key"])
        path_to_log = str(request_config["path_to_log"])
        preprocessor = Preprocessing()

        if is_xes:
            #preprocessor.import_event_log_xes(path , "case:concept:name", "concept:name", "time:timestamp")# hospital
            preprocessor.import_event_log_xes(path_to_log , case_id, activity, timestamp)# bpi 2019
        else:
            preprocessor.import_event_log_csv(path_to_log , case_id, activity, timestamp, ',')

        input_df = preprocessor.event_df

        cuda = True if request_config["cuda"]=="True" else False

        path_to_model = str(request_config["path_to_model"])

        config = Config()
        dic =  json.loads(request_config.get("config"))
        config.load_config(dic)

        nn_manager = NNManagement(config)
        nn_manager.import_nn_model(path_to_model)
        pm = PredictionManager(
            nn_manager.model, 
            case_id, 
            activity,
            timestamp, 
            config
        )

        time, event = pm.single_prediction_dataframe(input_df)
        print(time)
        print(event)
        return jsonify({
            "predicted_time":float(time), 
            "predicted_event":int(event)
        })


@routes.route('/random_search', methods = ["GET"])
def random_search():
    """
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
        sp = json.loads(request_config["search_params"])
        iterations= json.loads(request_config["iterations"])

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
        nn_manager.config.exponent = preprocessor.exponent

        nn_manager.random_search(
            train,
            test, 
            sp, 
            iterations, 
            preprocessor.case_id_key, 
            preprocessor.case_timestamp_key, 
            preprocessor.case_activity_key
        )

        config = nn_manager.config.asdict()

        model_path = request_config["model_name"]
        nn_manager.export_nn_model(model_path)

        with open(model_path, 'rb') as f:
            model_data = f.read()
        response = make_response(model_data)

        metadata = json.dumps({
            "config": config
        })

        # TODO: it might be convenient to also send the nn_mnager config.
        response.headers.set('Content-Type', 'application/octet-stream') # announce file included
        response.headers.set('Content-Disposition', 'attachment', filename='model.pt') 
        response.headers.set('X-Metadata', metadata) #include json metadata

        return response 






@routes.route('/grid_search', methods = ["GET"])
def grid_search():
    """
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
        sp = json.loads(request_config["search_params"])

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
        nn_manager.config.exponent = preprocessor.exponent

        nn_manager.grid_search(train, test, sp, preprocessor.case_id_key, preprocessor.case_timestamp_key, preprocessor.case_activity_key)

        config = nn_manager.config.asdict()

        model_path = request_config["model_name"]
        nn_manager.export_nn_model(model_path)

        with open(model_path, 'rb') as f:
            model_data = f.read()
        response = make_response(model_data)

        metadata = json.dumps({
            "config": config
        })

        # TODO: it might be convenient to also send the nn_mnager config.
        response.headers.set('Content-Type', 'application/octet-stream') # announce file included
        response.headers.set('Content-Disposition', 'attachment', filename='model.pt') 
        response.headers.set('X-Metadata', metadata) #include json metadata

        return response 






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
        case_id= str(request_config["case_id"])
        activity= str(request_config["activity_key"])
        timestamp= str(request_config["timestamp_key"])
        model_name= str(request_config["model_name"])

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
        nn_manager.config.exponent = preprocessor.exponent

        nn_manager.load_data(train, test, preprocessor.case_id_key, preprocessor.case_timestamp_key, preprocessor.case_activity_key)
        nn_manager.train()

        training_stats = nn_manager.get_training_statistics()
        config = nn_manager.config.asdict()

        nn_manager.export_nn_model(model_name)

        
        with open(model_name, 'rb') as f:
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

