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


@routes.route('/random_search', methods = ["POST"])
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
        "hidden_dim":[lower_bound,upper_bound] ,
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
        iterations = int(request_config["iterations"])
        sp = request_config["search_params"]
        split =  float(request_config["split"])

        for key in sp.keys():
            for i, val in enumerate(sp[key]): 
                sp[key][i] = int(val)

        preprocessor = Preprocessing()
        preprocessor.handle_import(is_xes, path_to_log, case_id, timestamp, activity)


        train, test= preprocessor.split_train_test(split)


        nn_manager= NNManagement() 


        nn_manager.config.cuda = True if request_config["cuda"] == "True" else False
        nn_manager.config = load_config_from_preprocessor(nn_manager.config, preprocessor) 
        
        
        acc= nn_manager.random_search(
            train,
            test, 
            sp, 
            iterations, 
            preprocessor.case_id_key, 
            preprocessor.case_timestamp_key, 
            preprocessor.case_activity_key
        )
        config = nn_manager.config.asdict()

        nn_manager.export_nn_model(request_config["model_path"])
        

        with open(f"{request_config['model_path']}.config.json", "w") as f:
            json.dump(config,f)
        data = {
            "acc":  acc
        }



        response = make_response(jsonify(data))




        return response
    




@routes.route('/grid_search', methods = ["POST"])
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
        "hidden_dim":[lower_bound,upper_bound, step] ,
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

        for key in sp.keys():
            for i, val in enumerate(sp[key]): 
                sp[key][i] = int(val)


        preprocessor = Preprocessing()
        preprocessor.handle_import(is_xes, path_to_log, case_id, timestamp, activity)


        train, test= preprocessor.split_train_test(float(request_config["split"]))


        nn_manager= NNManagement() 

        nn_manager.config.cuda = True if request_config["cuda"] == "True" else False

        nn_manager.config = load_config_from_preprocessor(nn_manager.config, preprocessor) 


        
        acc= nn_manager.grid_search(train, test, sp, preprocessor.case_id_key, preprocessor.case_timestamp_key, preprocessor.case_activity_key)
        
        config = nn_manager.config.asdict()

        nn_manager.export_nn_model(request_config["model_path"])
        

        with open(f"{request_config['model_path']}.config.json", "w") as f:
            json.dump(config,f)
        data = {
            "acc":  acc
        }

        response = make_response(jsonify(data))
        return response


@routes.route('/train_nn', methods = ["POST"])
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

        is_xes = request_config["is_xes"] 


        preprocessor = Preprocessing()
        preprocessor.handle_import(is_xes, path_to_log, case_id, timestamp, activity)


        train, test= preprocessor.split_train_test(float(request_config["split"]))


        nn_manager= NNManagement() 

        nn_manager.config.cuda = request_config["cuda"]
        nn_manager.config.seq_len = int(request_config["seq_len"])
        nn_manager.config.emb_dim= int(request_config["emb_dim"])
        nn_manager.config.hid_dim= int(request_config["hid_dim"])
        nn_manager.config.mlp_dim= int(request_config["mlp_dim"])
        nn_manager.config.lr= float(request_config["lr"])
        nn_manager.config.batch_size= int(request_config["batch_size"])
        nn_manager.config.epochs= int(request_config["epochs"])

        nn_manager.config = load_config_from_preprocessor(nn_manager.config, preprocessor) 

        nn_manager.load_data(train, test, case_id, timestamp, activity)

        nn_manager.train()

        training_stats = nn_manager.get_training_statistics()

        config = nn_manager.config.asdict()


        data = {
            "training_statistics": training_stats, 
        }

        nn_manager.export_nn_model(request_config["model_path"])


        with open(f"{request_config['model_path']}.config.json", "w") as f:
            json.dump(config,f)

        response = make_response(jsonify(data))

        return response

def load_config_from_preprocessor(config : Config, preprocessor : Preprocessing)-> Config:
    config.absolute_frequency_distribution = preprocessor.absolute_frequency_distribution
    config.number_classes = preprocessor.number_classes
    config.case_id_le = preprocessor.case_id_le
    config.activity_le = preprocessor.activity_le
    config.exponent = preprocessor.exponent
    config.case_activity_key = preprocessor.case_activity_key
    config.case_id_key = preprocessor.case_id_key
    config.case_timestamp_key = preprocessor.case_timestamp_key
    return config







@routes.route('/end_session')
def end_session():

    return ok

