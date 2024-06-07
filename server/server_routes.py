from flask import Blueprint, request, send_file, make_response, jsonify, Response
from server import preprocessing
from server import nn_manager
from server import process_model_manager
from server import prediction_manager
from server import exceptions
from functools import wraps


import base64
import os
import json
import torch

routes =Blueprint("Routes", __name__)










ok = {"status":"OK"}


def check_not_present_paths_factory(must_not_exist: list):
    def check_not_present_paths(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            data = request.get_json()
            for file in must_not_exist: 
                if os.path.isfile(data[file]):
                    print(data[file], "should NOT be present")
                    return {"error": "the target path for the new file already exists"},400
            return func( *args, **kwargs)
        return wrapper
    return check_not_present_paths



def check_required_paths_factory(must_be_present: list):
    def check_required_paths(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            data = request.get_json()
            for file in must_be_present: 
                if not os.path.isfile(data[file]):
                    print(data[file], "should be present")
                    return {"error": "one required path does not exist"},400

            return func( *args, **kwargs)
        return wrapper
    return check_required_paths 



def check_integers_factory(params: list):
    def check_integers(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            data = request.get_json()
            for param in params:
                try: 
                    i = int(data[param])
                except: 
                    print(param, "should be int")
                    return {"error": f"an integer param was set to another type"},400 
            return func( *args, **kwargs)
        return wrapper
    return check_integers 


def check_floats_factory(params: list):
    def check_floats(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            data = request.get_json()
            for param in params:
                try: 
                    i = float(data[param])
                except: 
                    print(param, "should be float")
                    return {"error": f"a float param was set to another type"},400 
            return func( *args, **kwargs)
        return wrapper
    return check_floats 



def check_booleans_factory(params: list): 
    def check_booleans(func): 
        @wraps(func)
        def wrapper(*args, **kwargs):
            data = request.get_json()
            for param in params:
                if not isinstance(data[param], bool):
                    print(param, "should be bool")
                    return {"error": f"a boolean param was set to another type"},400 
            return func( *args, **kwargs)
        return wrapper
    return check_booleans 






@routes.route('/start')
def start():
    return ok

@routes.route('/test')
def test():
    if torch.cuda.is_available():
        return {
            "CUDA available: ": torch.cuda.is_available(),
            "CUDA device count: ": torch.cuda.device_count(), 
            "CUDA device name: ": torch.cuda.get_device_name(0), 
        }
    elif not torch.cuda.is_available(): 
        return {
            "state": "no cuda"
        }
    return {
        "state": "error"
    }


@routes.route("/conformance", methods = ["POST"])
def conformance():
    
    if request.method == 'POST':
        request_config = request.args.to_dict()
        is_xes = True if request_config["is_xes"]=="True" else False
        case_id= str(request_config["case_id"])
        activity= str(request_config["activity_key"])
        timestamp= str(request_config["timestamp_key"])
        path_to_log = str(request_config["path_to_log"])
        petri_net_path  =  str(request_config.get("petri_net_path"))
        conformance_technique=  str(request_config.get("conformance_technique"))
        preprocessor = preprocessing.Preprocessing()
        preprocessor.handle_import(is_xes, path_to_log, case_id, timestamp, activity)
    
 
        with open(f"{petri_net_path}.json", "r") as f: 
            pn_config = json.load(f)

        pmm = process_model_manager.ProcessModelManager(
            preprocessor.event_df, 
            None, 
            None,
            preprocessor.case_activity_key,
            preprocessor.case_id_key,
            preprocessor.case_timestamp_key
        )
        pmm.initial_marking = pn_config["initial_marking"]
        pmm.final_marking= pn_config["final_marking"]
        pmm.load_petri_net(petri_net_path)
        pmm.unencoded_df = preprocessor.unencoded_df  #: generated automatically py preprocessor
        if conformance_technique == "token":
            fitness =pmm.conformance_checking_token_based_replay()
        else: 
            fitness =pmm.conformance_checking_alignments()
        return {"fitness": fitness}, 200

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


        preprocessor = preprocessing.Preprocessing()
        if is_xes:
            #preprocessor.import_event_log_xes(path , "case:concept:name", "concept:name", "time:timestamp")# hospital
            preprocessor.import_event_log_xes(path_to_log , case_id, activity, timestamp)# bpi 2019
        else:
            preprocessor.import_event_log_csv(path_to_log , case_id, activity, timestamp, ',')

        config = nn_manager.Config()
        config.load_config(dic)
        neural_manager = nn_manager.NNManagement(config)
        neural_manager.import_nn_model(path_to_model)

        pmm = preprocessing.ProcessModelManager(
            preprocessor.event_df, 
            neural_manager.model, 
            neural_manager.config,
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


        preprocessor = preprocessing.Preprocessing()
        if is_xes:
            #preprocessor.import_event_log_xes(path , "case:concept:name", "concept:name", "time:timestamp")# hospital
            preprocessor.import_event_log_xes(path_to_log , case_id, activity, timestamp)# bpi 2019
        else:
            preprocessor.import_event_log_csv(path_to_log , case_id, activity, timestamp, ',')


        config = nn_manager.Config()
        config.load_config(dic)

        neural_manager = nn_manager.NNManagement(config)
        neural_manager.import_nn_model(path_to_model)

        pmm = process_model_manager.ProcessModelManager(
            preprocessor.event_df, 
            neural_manager.model, 
            neural_manager.config,
            preprocessor.case_activity_key,
            preprocessor.case_id_key,
            preprocessor.case_timestamp_key
        )
        pmm.end_activities = preprocessor.find_end_activities()



        #: TODO: check the combinations for the bool attributes, some combinations 
        # are not well defined.
        pmm.generate_predictive_log(non_stop=non_stop, upper =upper, random_cuts=random_cuts,cut_length=cut_length, new_log_path = new_log_path )

        
        
        
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
        preprocessor = preprocessing.Preprocessing()

        if is_xes:
            #preprocessor.import_event_log_xes(path , "case:concept:name", "concept:name", "time:timestamp")# hospital
            preprocessor.import_event_log_xes(path_to_log , case_id, activity, timestamp)# bpi 2019
        else:
            preprocessor.import_event_log_csv(path_to_log , case_id, activity, timestamp, ',')

        input_df = preprocessor.event_df

        cuda = True if request_config["cuda"]=="True" else False

        path_to_model = str(request_config["path_to_model"])

        config = nn_manager.Config()
        dic =  json.loads(request_config.get("config"))
        config.load_config(dic)

        neural_manager = nn_manager.NNManagement(config)
        neural_manager.import_nn_model(path_to_model)
        pm = prediction_manager.PredictionManager(
            neural_manager.model, 
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
        preprocessor = preprocessing.Preprocessing()

        if is_xes:
            #preprocessor.import_event_log_xes(path , "case:concept:name", "concept:name", "time:timestamp")# hospital
            preprocessor.import_event_log_xes(path_to_log , case_id, activity, timestamp)# bpi 2019
        else:
            preprocessor.import_event_log_csv(path_to_log , case_id, activity, timestamp, ',')

        input_df = preprocessor.event_df

        cuda = True if request_config["cuda"]=="True" else False

        path_to_model = str(request_config["path_to_model"])

        config = nn_manager.Config()
        dic =  json.loads(request_config.get("config"))
        config.load_config(dic)

        neural_manager = nn_manager.NNManagement(config)
        neural_manager.import_nn_model(path_to_model)
        pm = prediction_manager.PredictionManager(
            neural_manager.model, 
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


@routes.route('/random_search', methods = ["POST"])
@check_booleans_factory(["cuda", "is_xes"])
@check_integers_factory(["seq_len","batch_size", "epochs", "iterations"])
@check_floats_factory(["lr", "split"])
@check_required_paths_factory(["path_to_log"])
@check_not_present_paths_factory(["model_path"])
def random_search():
    """
    carries out random search. It only accepts post requests. 

    the following data is expected in the JSON body of the request: 
    :param path_to_log: path to the log used for training. must not be encoded
    :param split: float in range [0,1]. represents train - test ratio. 
    :param case_id: name of case id column
    :param activity_key: name of activity column
    :param timestamp_key: name of timestamp column
    :param cuda: True/False if cuda used/not used. 
    :param seq_len: length of the sliding window used. 
    :param lr: learning rate
    :param batch_size: batch size 
    :param epochs: number of epochs
    :param is_xes: is the log in xes format?
    :param iterations: number of iterations for random search.
    :param search_params: dictioary of the format: 
    ```py
    {
        "hid_dim":[lower_bound,upper_bound] ,
        "mlp_dim":[lower_bound, upper_bound] ,
        "emb_dim":[lower_bound, upper_bound] 
    }
    ```

    the response contains the following and has the next side effects: 
    a json document is returned with the foloing parameters: 
    :return config: the config file that is used for process prophet. 
    :return acc: the best accuracy of training achieved
    :return model: a base64 encoded pt file containing the model setup  ready
    for importing
    """
    if request.method == 'POST':
        request_config = request.get_json()
        

        is_xes = request_config["is_xes"] 

        path_to_log = str(request_config["path_to_log"])
        case_id= str(request_config["case_id"])
        activity= str(request_config["activity_key"])
        timestamp= str(request_config["timestamp_key"])
        split = float(request_config["split"])
        iterations = int(request_config["iterations"])

        sp = request_config["search_params"]


        sp_keys= sp.keys()
        if "hid_dim" not in sp_keys or "mlp_dim" not in sp_keys or "emb_dim" not in sp_keys:
            return {"error": "missing key in search params"},400 
        for key in ["hid_dim", "mlp_dim", "emb_dim"]:
            if len(sp[key])!=2: 
                    return {"error": f"search param(s) missing"},400 
            for i, val in enumerate(sp[key]): 
                try:
                    sp[key][i] = int(val)
                except: 
                    return {"error": f"non integer found in search params"},400 


        preprocessor = preprocessing.Preprocessing()


        try: 
            preprocessor.handle_import(is_xes, path_to_log, case_id, timestamp, activity)
        except Exception as e: 
            return {
                "error":str(e), 
                "description": "error while importing"
            }, 400

        try:
            train, test= preprocessor.split_train_test(float(request_config["split"]))
        except exceptions.TrainPercentageTooHigh: 
            return {"error": "train percentage must be in range (0,1) and should not yield empty sublogs"}, 400
        except Exception as e: 
            return {
                "error":str(e),
                "description": "error while importing"
            }, 400

     
        neural_manager= nn_manager.NNManagement() 


        neural_manager.config.cuda =  request_config["cuda"]
        neural_manager.config = load_config_from_preprocessor(neural_manager.config, preprocessor) 
        
        neural_manager.load_data(train, test, preprocessor.case_id_key, preprocessor.case_timestamp_key, preprocessor.case_activity_key)
        
        try:
            acc= neural_manager.random_search(sp, iterations)
        except exceptions.NaNException as e: 
            return {
                "error": str(e)
            }
        except Exception as e: 
            return {
                "error":str(e),
                "description": "error while training"
            }, 400
        
        
        config = neural_manager.config.asdict()

        neural_manager.export_nn_model(request_config["model_path"])
        

        with open(f"{request_config['model_path'][:-3]}.config.json", "w") as f:
            json.dump(config,f)
        data = {
            "acc":  acc
        }



        response = make_response(jsonify(data))




        return response
    




@routes.route('/grid_search', methods = ["POST"])
@check_booleans_factory(["cuda", "is_xes"])
@check_integers_factory(["seq_len","batch_size", "epochs"])
@check_floats_factory(["lr", "split"])
@check_required_paths_factory(["path_to_log"])
@check_not_present_paths_factory(["model_path"])
def grid_search():
    """
    carries out grid search. It only accepts post requests. 

    the following data is expected in the JSON body of the request: 
    :param path_to_log: path to the log used for training. must not be encoded
    :param split: float in range [0,1]. represents train - test ratio. 
    :param case_id: name of case id column
    :param activity_key: name of activity column
    :param timestamp_key: name of timestamp column
    :param cuda: True/False if cuda used/not used. 
    :param seq_len: length of the sliding window used. 
    :param lr: learning rate
    :param batch_size: batch size 
    :param epochs: number of epochs
    :param is_xes: is the log in xes format?
    :param search_params: dictioary of the format: 
    ```py
    {
        "hid_dim":[lower_bound,upper_bound, step] ,
        "mlp_dim":[lower_bound, upper_bound, step] ,
        "emb_dim":[lower_bound, upper_bound, step] 
    }
    ```
    the response contains the following and has the next side effects: 
    a json document is returned with the foloing parameters: 
    :return config: the config file that is used for process prophet. 
    :return acc: the best accuracy of training achieved
    :return model: a base64 encoded pt file containing the model setup  ready
    for importing
    """
    if request.method == 'POST':
        request_config = request.get_json()
        
        
        is_xes = request_config["is_xes"] 
        path_to_log = str(request_config["path_to_log"])
        case_id= str(request_config["case_id"])
        activity= str(request_config["activity_key"])
        timestamp= str(request_config["timestamp_key"])
        sp = request_config["search_params"]
        split = float(request_config["split"])


      

        sp_keys= sp.keys()
        if "hid_dim" not in sp_keys or "mlp_dim" not in sp_keys or "emb_dim" not in sp_keys:
            return {"error": f"missing key in search params"},400 

        for key in ["hid_dim", "mlp_dim", "emb_dim"]: #:just in case more keys are present.
            if len(sp[key])!=3: 
                    return {"error": f"search param(s) missing"},400 
            for i, val in enumerate(sp[key]): 
                try:
                    sp[key][i] = int(val)
                except: 
                    return {"error": f"non integer found in search params"},400 



        preprocessor = preprocessing.Preprocessing()
        try: 
            preprocessor.handle_import(is_xes, path_to_log, case_id, timestamp, activity)
        except Exception as e: 
            return {
                "error":str(e), 
                "description": "error while importing"
            }, 400

        try:
            train, test= preprocessor.split_train_test(split)
        except exceptions.TrainPercentageTooHigh: 
            return {"error": "train percentage must be in range (0,1) and should not yield empty sublogs"}, 400
        except Exception as e: 
            return {
                "error":str(e),
                "description": "error while importing"
            }, 400


        neural_manager= nn_manager.NNManagement() 

        neural_manager.config.cuda =  request_config["cuda"]

        neural_manager.config = load_config_from_preprocessor(neural_manager.config, preprocessor) 

        neural_manager.load_data(train, test, preprocessor.case_id_key, preprocessor.case_timestamp_key, preprocessor.case_activity_key)
        

        try:
            acc= neural_manager.grid_search(sp)
        except exceptions.NaNException as e: 
            return {
                "error": str(e)
            }
        except Exception as e: 
            return {
                "error":str(e), 
                "description": "error while training"
            }, 400
        
        config = neural_manager.config.asdict()

        neural_manager.export_nn_model(request_config["model_path"])
        

        with open(f"{request_config['model_path'][:-3]}.config.json", "w") as f:
            json.dump(config,f)
        data = {
            "acc":  acc
        }

        response = make_response(jsonify(data))
        return response

@routes.route('/train_nn', methods = ["POST"])
@check_booleans_factory(["cuda", "is_xes"])
@check_integers_factory(["seq_len", "emb_dim", "hid_dim", "mlp_dim", "batch_size", "epochs"])
@check_floats_factory(["lr", "split"])
@check_required_paths_factory(["path_to_log"])
@check_not_present_paths_factory(["model_path"])
def train_nn():
    """
    trains the RMTPP neural network. 

    the following data is expected in the JSON body of the request: 
    :param path_to_log: path to the log used for training. must not be encoded
    :param split: float in range [0,1]. represents train - test ratio. 
    :param case_id: name of case id column
    :param activity_key: name of activity column
    :param timestamp_key: name of timestamp column
    :param cuda: True/False if cuda used/not used. 
    :param seq_len: length of the sliding window used. 
    :param lr: learning rate
    :param batch_size: batch size 
    :param epochs: number of epochs
    :param is_xes: is the log in xes format?
    :param emb_dim: embedding dimension
    :param hid_dim: hidden layer dimension
    :param mlp_dim: mlp dimension
    
    the response contains the following and has the next side effects: 
    a json document is returned with the foloing parameters: 
    :return config: the config file that is used for process prophet. 
    :return acc: the training accuracy achieved
    :return model: a base64 encoded pt file containing the model setup  ready
    for importing
    """
    if request.method == 'POST':
        request_config = request.get_json()
        path_to_log = str(request_config["path_to_log"])
        case_id= str(request_config["case_id"])
        activity= str(request_config["activity_key"])
        timestamp= str(request_config["timestamp_key"])




        preprocessor = preprocessing.Preprocessing()
        
        path = request_config["model_path"]
        if os.path.isfile(request_config["model_path"]):
            return {"error": f"{request_config["model_path"]} model already exists..."},400 

        path = request_config["path_to_log"]
        if not os.path.isfile(request_config["path_to_log"]):
            return {"error": f"{request_config["path_to_log"]} does not exist..."},400 

        try: 
            preprocessor.handle_import(request_config['is_xes'], path_to_log, case_id, timestamp, activity)
        except Exception as e: 
            return {
                "error":str(e), 
                "description": "error while importing"
            }, 400

        try:
            train, test= preprocessor.split_train_test(float(request_config["split"]))
        except exceptions.TrainPercentageTooHigh: 
            return {"error": "train percentage must be in range (0,1) and should not yield empty sublogs"}, 400
        except Exception as e: 
            return {
                "error":str(e),
                "description": "error while importing"
            }, 400


        neural_manager= nn_manager.NNManagement() 

        neural_manager.config = load_config_from_params(neural_manager.config, request_config)

        neural_manager.config = load_config_from_preprocessor(neural_manager.config, preprocessor) 

        neural_manager.load_data(train, test, case_id, timestamp, activity)

        try:
            neural_manager.train()
        except exceptions.NaNException as e: 
            return {
                "error": str(e)
            }
        except Exception as e: 
            return {
                "error":str(e),
                "description": "error while training"
            }, 400



        training_stats = neural_manager.get_training_statistics()

        config = neural_manager.config.asdict()

        data = {
            "training_statistics": training_stats, 
        }

        neural_manager.export_nn_model(request_config["model_path"])


        with open(f"{request_config['model_path'][:-3]}.config.json", "w") as f:
            json.dump(config,f)

        response = make_response(jsonify(data))

        return response




def load_config_from_params(config: nn_manager.Config, request_config:dict) -> nn_manager.Config:
        config.cuda = request_config["cuda"]
        config.seq_len = int(request_config["seq_len"])
        config.emb_dim= int(request_config["emb_dim"])
        config.hid_dim= int(request_config["hid_dim"])
        config.mlp_dim= int(request_config["mlp_dim"])
        config.lr= float(request_config["lr"])
        config.batch_size= int(request_config["batch_size"])
        config.epochs= int(request_config["epochs"])
        return config



def load_config_from_preprocessor(config : nn_manager.Config, preprocessor : preprocessing.Preprocessing)-> nn_manager.Config:
    config.absolute_frequency_distribution = preprocessor.absolute_frequency_distribution
    config.number_classes = preprocessor.number_classes
    config.case_id_le = preprocessor.case_id_le
    config.activity_le = preprocessor.activity_le
    config.exponent = preprocessor.exponent
    config.case_activity_key = preprocessor.case_activity_key
    config.case_id_key = preprocessor.case_id_key
    config.case_timestamp_key = preprocessor.case_timestamp_key
    return config




@routes.route('/replace_with_mode', methods = ["POST"])
@check_booleans_factory(["is_xes"])
@check_required_paths_factory(["path_to_log"])
@check_not_present_paths_factory(["save_path"])
def replace_with_mode():
    """
    replaces NaN's in the activity column with median
    the following data is expected in the JSON body of the request: 
    :param path_to_log: path to the log used for training. must not be encoded
    :param case_id: name of case id column
    :param activity_key: name of activity column
    :param timestamp_key: name of timestamp column
    :param save_path: path where the processed event log is exported 
    """
    if request.method == 'POST':
        request_config = request.get_json()


        
        is_xes = request_config["is_xes"] 
        path_to_log = str(request_config["path_to_log"])
        case_id= str(request_config["case_id"])
        activity= str(request_config["activity_key"])
        timestamp= str(request_config["timestamp_key"])
        save_path= str(request_config["save_path"])
        if not is_xes: 
            sep= str(request_config["sep"])
        else: 
            sep = ""


        preprocessor = preprocessing.Preprocessing()
        try: 
            preprocessor.handle_import(is_xes, path_to_log, case_id, timestamp, activity,sep=sep, formatting=False )
        except Exception as e: 
            return {
                "error":str(e),
                "description": "error while importing"
            }, 400

        success= preprocessor.replace_activity_nan_with_mode()
        if success:
            preprocessor.event_df.to_csv(save_path, sep = ",")
            return {
                "status": "successfully finished", 
                "save_path":save_path
                }, 200
        return {"error": "nan replacement went wrong..."}, 400





@routes.route('/add_unique_start_end', methods = ["POST"])
@check_booleans_factory(["is_xes"])
@check_required_paths_factory(["path_to_log"])
@check_not_present_paths_factory(["save_path"])
def add_unique_start_end():
    """
    adds a unique start/end activity to the log
    the following data is expected in the JSON body of the request: 
    :param path_to_log: path to the log used for training. must not be encoded
    :param case_id: name of case id column
    :param activity_key: name of activity column
    :param timestamp_key: name of timestamp column
    :param save_path: path where the processed event log is exported 
    """
    if request.method == 'POST':
        request_config = request.get_json()

        is_xes = request_config["is_xes"] 
        path_to_log = str(request_config["path_to_log"])
        case_id= str(request_config["case_id"])
        activity= str(request_config["activity_key"])
        timestamp= str(request_config["timestamp_key"])
        save_path= str(request_config["save_path"])

        if not is_xes: 
            sep= str(request_config["sep"])
        else: 
            sep = "" 

        preprocessor = preprocessing.Preprocessing()

        try: 
            preprocessor.handle_import(is_xes, path_to_log, case_id, timestamp, activity,sep=sep, formatting=False )
        except Exception as e: 
            return {
                "error":str(e),
                "description": "error while importing"
            }, 400


        success= preprocessor.add_unique_start_end_activity()
        if success:
            preprocessor.event_df.to_csv(save_path, sep = ",")
            return {
                "status": "successfully created", 
                "save_path":save_path
                }, 200
        return {"error": "operation not necessary, the log already has a unique start/end activity"}, 400

@routes.route('/remove_duplicates', methods = ["POST"])
@check_booleans_factory(["is_xes"])
@check_required_paths_factory(["path_to_log"])
@check_not_present_paths_factory(["save_path"])
def remove_duplicates():
    """
    removes the duplicates ie the rows where the same activity happened at the same time in the same case id.

    the following data is expected in the JSON body of the request: 
    :param path_to_log: path to the log used for training. must not be encoded
    :param case_id: name of case id column
    :param activity_key: name of activity column
    :param timestamp_key: name of timestamp column
    :param save_path: path where the processed event log is exported 
    """
    if request.method == 'POST':
        request_config = request.get_json()

        is_xes = request_config["is_xes"] 


        path_to_log = str(request_config["path_to_log"])
        case_id= str(request_config["case_id"])
        activity= str(request_config["activity_key"])
        timestamp= str(request_config["timestamp_key"])
        save_path= str(request_config["save_path"])

        if not is_xes: 
            sep= str(request_config["sep"])
        else:
            sep = ""

        preprocessor = preprocessing.Preprocessing()
        try: 
            preprocessor.handle_import(is_xes, path_to_log, case_id, timestamp, activity,sep=sep, formatting=False )
        except Exception as e: 
            return {
                "error":str(e),
                "description": "error while importing"
            }, 400


        success= preprocessor.remove_duplicate_rows()
        if success:
            path = save_path
            preprocessor.event_df.to_csv(save_path, sep = ",")
            return {"save_path":save_path}, 200
        return {"error": "removing duplicates went wrong..."},400 





