from flask import Blueprint, request, send_file, make_response, jsonify, Response
from preprocessing import * 
from nn_manager import *
from process_model_manager  import *
from prediction_manager import *
import base64
import os

routes =Blueprint("Routes", __name__)


ok = {"status":"OK"}

@routes.route('/start')
def start():
    return ok


@routes.route('/generate_predictive_process_model', methods = ["GET"])
def generate_predictive_process_model():
    
    if request.method == 'GET':
        request_config = request.args.to_dict()
        is_xes = True if request_config["is_xes"]=="True" else False
        case_id= str(request_config["case_id"])
        activity= str(request_config["activity_key"])
        timestamp= str(request_config["timestamp_key"])
        path_to_log = str(request_config["path_to_log"])
        path_to_model = str(request_config["path_to_model"])
        cuda = True if request_config["cuda"]=="True" else False
        dic =  json.loads(request_config.get("config"))
        new_log_path=  request_config.get("new_log_path")
        selected_model  =  str(request_config.get("selected_model"))
        petri_net_path  =  str(request_config.get("petri_net_path"))

        minig_algo_config=  json.loads(request_config.get("mining_algo_config"))


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
        pmm.import_predictive_df(new_log_path)
        match selected_model: 
            case "alpha_miner":
                pmm.alpha_miner(petri_net_path)
            case "heuristic_miner":
                pmm.heuristic_miner(
                    petri_net_path, 
                    minig_algo_config["dependency_threshold"], 
                    minig_algo_config["and_threshold"], 
                    minig_algo_config["loop_two_threshold"]
                )
            case "inductive_miner":
                pmm.inductive_miner(petri_net_path, minig_algo_config["noise_threshold"])
            case "prefix_tree_miner":
                pmm.prefix_tree_miner(petri_net_path)


        
        return ok #they are already encoded




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






@routes.route('/train_nn', methods = ["POST"])
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
    if request.method == 'POST':
        request_config = request.get_json()
        path_to_log = str(request_config["path_to_log"])
        case_id= str(request_config["case_id"])
        activity= str(request_config["activity_key"])
        timestamp= str(request_config["timestamp_key"])
        model_path= str(request_config["model_path"])

        is_xes = request_config["is_xes"] 


        preprocessor = Preprocessing()
        preprocessor.handle_import(is_xes, path_to_log, case_id, timestamp, activity)


        train, test= preprocessor.split_train_test(float(request_config["split"]))


        nn_manager= NNManagement() 

        nn_manager.config.cuda = True if request_config["cuda"] == "True" else False
        nn_manager.config.absolute_frequency_distribution = preprocessor.absolute_frequency_distribution
        nn_manager.config.number_classes = preprocessor.number_classes
        nn_manager.config.activity_le = preprocessor.activity_le
        nn_manager.config.case_id_le = preprocessor.case_id_le
        nn_manager.load_data(train, test, case_id, timestamp, activity)
        nn_manager.train()

        training_stats = nn_manager.get_training_statistics()
 
        config = nn_manager.config.asdict()


        trash_path = "temp/model.pt"
        nn_manager.export_nn_model(trash_path)
        
        
        data = {
            "training_statistics": training_stats, 
            "config": config
        }
        with open(trash_path, 'rb') as f:
            model_data = f.read()



        encoded_file_content = base64.b64encode(model_data).decode('utf-8')
        data["file_content"] = encoded_file_content
        data["file_name"] = "model.pt" 


        response = make_response(jsonify(data))

        response.headers['Content-Disposition']=  'attachment; filename=model.pt'
        response.headers['Content-Type']=  'application/json'
        if not os.path.exists(trash_path):
            return make_response(jsonify({"error": "File not found"}), 404)
        os.remove(trash_path)



        return response




@routes.route('/import_log', methods = ["GET"])
def import_log():
    """
    :param path_to_log: the program will only search the path ../projects/<subfolder>/<file_name> 
    so the extected addr is of the form `/subfolder/file_name`.
    :param is_xes: Boolean expected. if False, then csv expected.
    :param export path: path where the preprocessed event log is exported.
    :param case_id: case id column name. just used if is_xes is False
    :param activity_key: activity column name. just used if is_xes is False
    """
    if request.method == 'GET':
        request_config = request.args.to_dict()
        is_xes = True if request_config["is_xes"]=="True" else False
        path_to_log = str(request_config["path_to_log"])
        case_id= str(request_config["case_id"])
        activity= str(request_config["activity_key"])
        timestamp= str(request_config["timestamp_key"])
        export_path= str(request_config["export_path"])

        print(path_to_log)
        print(export_path)
        preprocessor = Preprocessing()
        preprocessor.handle_import(is_xes, path_to_log, case_id, timestamp, activity)
        
        
        #: TODO: this should not write, but instead send a file.
        preprocessor.event_df.to_csv(export_path, sep = ",")

        config = Config()
        config.absolute_frequency_distribution = preprocessor.absolute_frequency_distribution
        config.number_classes = preprocessor.number_classes
        config.case_id_le = preprocessor.case_id_le
        config.activity_le = preprocessor.activity_le
        config.exponent = preprocessor.exponent
        config.case_activity_key = preprocessor.case_activity_key
        config.case_id_key = preprocessor.case_id_key
        config.case_timestamp_key = preprocessor.case_timestamp_key
        configuration_dict = config.asdict()
        with open(f"{export_path}.json", 'w') as f: 
            json.dump( configuration_dict, f)



        
        return ok




@routes.route('/end_session')
def end_session():

    return ok

