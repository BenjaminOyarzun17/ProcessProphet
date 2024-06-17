"""
This file contains all the supported server routes by ProcessProphet. Supported routes: 
- `/remove_duplicates`
- `/add_unique_start_end`
- `/replace_with_mode`
- `/train_nn`
- `/grid_search`
- `/random_search`
- `/single_prediction`
- `/multiple_prediction`
- `k/generate_predictive_log`
- `/generate_predictive_process_model`
- `/conformance`
"""


from flask import Blueprint, request, make_response, jsonify, Response
from server import preprocessing
from server import nn_manager
from server import process_model_manager
from server import prediction_manager
from server import exceptions
from functools import wraps
import os
import json
import torch

routes =Blueprint("Routes", __name__)










ok = {"status":"OK"}




def check_not_present_paths_factory(must_not_exist: list):
    """
    this decorator checks in the given file paths list if each file does not exist
    if it does, an error is sent as response (the user should know the input is wrong) 
    """
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
    """
    this decorator checks in the given file paths list if each file does exist. 
    very useful for checking if a log exists for example.
    if it does not, an error is sent as response  (the user should know the input is wrong)
    """
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
    """
    all parameters in the given list are checked whether they are of integer type
    otherwise an error response is sent.  (the user should know the input is wrong)
    """
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
    """
    all parameters in the given list are checked whether they are of float type
    otherwise an error response is sent.  (the user should know the input is wrong)
    """
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
    """
    all parameters in the given list are checked whether they are of bool type
    otherwise an error response is sent.  (the user should know the input is wrong)
    """
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
    #: just a testing route
    return ok, 200

@routes.route('/test')
def test():
    #: used for testing cuda availability. very recommended before using cuda; sometimes
    # docker configuration breaks the cuda settings. 
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
@check_required_paths_factory(['path_to_log', "petri_net_path"])
def conformance():
    """
    path for conformance checking. the following parameters are expected: 

    :param is_xes: whether the input log is xes or not (otherwise csv). 
    :param case_id: case id column name
    :param activity_key: activity column name
    :param timestamp: timestamp column name
    :param path_to_log: path to the event log 
    :param petri_net_path:  path to the petri net used for conf. checking.
    :param conformance_technique:  either "token" or "alignment". this selects the corresponding conf. checking technique. 
    :return fitness: the achieved fitness by the model. 
    """
    
    if request.method == 'POST':
        request_config = request.get_json()

        #: extract the request params
        is_xes = request_config["is_xes"]
        case_id= str(request_config["case_id"])
        activity= str(request_config["activity_key"])
        timestamp= str(request_config["timestamp_key"])

        path_to_log = str(request_config['path_to_log'])
        petri_net_path  =  str(request_config["petri_net_path"])

        conformance_technique=  str(request_config["conformance_technique"])

        preprocessor = preprocessing.Preprocessing()

        #: start by importing the log used for conf checking
        try:
            preprocessor.handle_import(is_xes, path_to_log, case_id, timestamp, activity)
        except Exception as e: 
            #: return an error and a description if something goes wrong
            return {"error": str(e)}, 400
 
        #: import the petri net configuration (contains the start, end markings for example)
        with open(f"{petri_net_path}.json", "r") as f: 
            pn_config = json.load(f)

        #: prepare the process model manager intance for conf. checking
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

        #: run the conformance checking technique. 
        try:
            if conformance_technique == "token":
                fitness =pmm.conformance_checking_token_based_replay()
            else: 
                fitness =pmm.conformance_checking_alignments()
        except Exception as e: 
            #: return an error and a description if something goes wrong
            return {"error": str(e)}, 400

        #: return the fitness
        return {"fitness": fitness}, 200

@routes.route('/generate_predictive_process_model', methods = ["POST"])
@check_required_paths_factory(['path_to_log', "config"])
@check_not_present_paths_factory(["petri_net_path"])
def generate_predictive_process_model():
    """
    path for creating a predictive process model ie a petri net. 


    the petri net and its config are exported in the given path

    :param is_xes: whether the input log is xes or not (otherwise csv). 
    :param case_id: case id column name
    :param activity_key: activity column name
    :param timestamp: timestamp column name
    :param path_to_log: path to the event log 
    :param petri_net_path:  path where the pnml file and the json file should be exported. 
    :param selected_model:  selected minign model ("alpha_miner", "heuristic_miner" , "inductive_miner", "prefix_tree_miner")
    :param mining_algo_config: settings for the selected process mining algorithm
    :param sep: column separator (used for csv files)
    :param config: path to the config file for the model
    """
    if request.method == 'POST':
        #: extract the params
        request_config = request.get_json()

        case_id= str(request_config["case_id"])
        activity= str(request_config["activity_key"])
        timestamp= str(request_config["timestamp_key"])

        path_to_log = str(request_config['path_to_log'])
        selected_mining_algo  =  str(request_config["selected_model"])
        petri_net_path  =  str(request_config["petri_net_path"])

        minig_algo_config=  request_config["mining_algo_config"]
        sep = request_config["sep"]

        #: import the predictive log  (ie with partial traces completed with predictions)
        preprocessor = preprocessing.Preprocessing()
        try:
            preprocessor.handle_import(False,path_to_log,case_id, timestamp,activity, sep= sep)
        except Exception as e:
            #: if something goes wrong...
            return {"error": str(e)}, 400



        #: load the config for the model
        config = nn_manager.Config()
        with open(request_config["config"], "r") as f: 
            dic = json.load(f)
        config.load_config(dic)
       

        #: initialize the process model manager intance used for petri net generation
        pmm = process_model_manager.ProcessModelManager(
            preprocessor.event_df, 
            None, 
            config,
            preprocessor.case_activity_key,
            preprocessor.case_id_key,
            preprocessor.case_timestamp_key
        )

        pmm.end_activities = preprocessor.find_end_activities()

        #: load the predictive df 
        pmm.import_predictive_df(path_to_log)

        #: runa mining algorithm. file saving is handled by each miner function 
        #: the exports is handled by each miner function, since a pmp4py call is necessary. 
        #: therefore it makes more sense to use the wrapper classses (managers) instead. 
        try:
            match selected_mining_algo: 
                case "alpha_miner":
                    pmm.alpha_miner(petri_net_path)
                case "heuristic_miner":
                    try:
                        dependency_thr = float(minig_algo_config["dependency_threshold"])
                        and_thr = float(minig_algo_config["and_threshold"])
                        loop_thr =float(minig_algo_config["loop_two_threshold"])
                    except:
                        print(e)
                        return {"error": f"a float param was set to another type"},400 
                    pmm.heuristic_miner(
                            petri_net_path, 
                            dependency_threshold= dependency_thr, 
                            and_threshold=and_thr, 
                            loop_two_threshold=loop_thr 
                    )
                case "inductive_miner":
                    try:
                        noise_thr = float(minig_algo_config["noise_threshold"])
                    except:
                        return {"error": f"a float param was set to another type"},400 
                    pmm.inductive_miner(petri_net_path,noise_thr )
                case "prefix_tree_miner":
                    pmm.prefix_tree_miner(petri_net_path)
        except Exception as e:
            print(e)
            return {"error": str(e)}, 400
            #: if something goes wrong, return the error

        
        #: save the initial and final markings int he json file. 
        initial = str(pmm.initial_marking)
        final  = str(pmm.final_marking)
        petri_net_config = {
            "initial_marking": initial,
            "final_marking":    final 
        }
        with open (f"{petri_net_path}.json", "w") as f:
            json.dump(petri_net_config,f)
        
        return ok, 200 #the 200 here is important




@routes.route('/generate_predictive_log', methods = ["POST"])
@check_required_paths_factory(['path_to_log', "config", "path_to_model"])
@check_not_present_paths_factory(["new_log_path"])
@check_integers_factory(["upper", "cut_length"])
@check_booleans_factory(["non_stop","is_xes", "random_cuts"])
def generate_predictive_log():
    """
    generates the predictive event log by cutting all traces using the given configuration
    and by exteniding this cut traces with predictions.
    
    the predictive log is exported in the given path

    :param is_xes: whether the input log is xes or not (otherwise csv). 
    :param case_id: case id column name
    :param activity_key: activity column name
    :param timestamp: timestamp column name
    :param path_to_log: path to the event log used for cutting
    :param path_to_model: path to the RNN model used for making predictions 
    :param new_log_path: path where the predictive log should be saved (csv format is default.) 
    :param sep: column separator (used for csv input logs)
    :param config: path to the config file for the model
    :param random_cuts: boolean. if set to true, each trace is cut at a random sequence index. 
    :param non_stop: boolean. if set to true, predictions are made until an end marking is reached. 
    :param cut_length: in case of random cuts = non_stop = False, we cut from the tail of each trace 
    the last `cut_length` events. 
    :param upper: upper bound for the number of iterations the non_stop variant should run (just for safety)
    """
    if request.method == 'POST':
        request_config = request.get_json()

        #: get the data
        case_id= str(request_config["case_id"])
        activity= str(request_config["activity_key"])
        timestamp= str(request_config["timestamp_key"])
        
        path_to_log = str(request_config['path_to_log'])
        path_to_model = str(request_config["path_to_model"])
        new_log_path=  request_config["new_log_path"]

        is_xes = request_config["is_xes"]
        non_stop = bool(request_config["non_stop"])
        random_cuts =bool(request_config["random_cuts"])

        sep=  request_config.get("sep")

        cut_length = int(request_config["cut_length"])
        upper = int(request_config["upper"])

        preprocessor = preprocessing.Preprocessing()

        #: import the log for cutitng
        try: 
            preprocessor.handle_import(is_xes, path_to_log, case_id, timestamp, activity,sep=sep, formatting=True )
        except Exception as e: 
            #:  notify the error 
            return {"error": str(e)}, 400


        #: load RNN config file and model
        config = nn_manager.Config()
        with open(request_config["config"], "r") as f: 
            dic = json.load(f)
        config.load_config(dic)
        neural_manager = nn_manager.NNManagement(config)
        neural_manager.import_nn_model(path_to_model)
        

        #: initialize proces model manager and generate the prdictive log
        # the pmm is in charge of exporing the log.
        try:
            pmm = process_model_manager.ProcessModelManager(
                preprocessor.event_df, 
                neural_manager.model, 
                neural_manager.config,
                preprocessor.case_activity_key,
                preprocessor.case_id_key,
                preprocessor.case_timestamp_key
            )
            pmm.end_activities = preprocessor.find_end_activities()
            pmm.generate_predictive_log(non_stop=non_stop, upper =upper, random_cuts=random_cuts,cut_length=cut_length, new_log_path = new_log_path )
        except Exception as e: 
            #: notify errors
            return {"error": str(e)}, 400
        
        
        return ok, 200 











@routes.route('/multiple_prediction', methods = ["POST"])
@check_required_paths_factory(['path_to_log', "config", "path_to_model"])
@check_not_present_paths_factory(['prediction_file_name'])
@check_integers_factory(["degree", "depth"])
def multiple_prediction():
    """
    given a partial trace carry out multiple predictions.  
  
    generates a file containing the multiple predicitons in the given path.

    :param case_id: case id column name
    :param activity_key: activity column name
    :param timestamp: timestamp column name
    :param path_to_log: path to the input partial trace. must contain a single case id and columns 
    must have the same names as the the ones used in the log for training. it must be a csv file with "," as separator
    :param path_to_model: path to the RNN model used for making predictions 
    :param prediction_file_name: file name for the output file that will contain the predictions
    :param config: path to the config file for the model
    :param degree: branching degree of the generated prediction tree
    :param depth: depth that the predictive tree should have
    """

    if request.method == 'POST':
        request_config = request.get_json()
        #: extract params
        depth = int(request_config["depth"])
        degree = int(request_config["degree"])
        case_id= str(request_config["case_id"])
        activity= str(request_config["activity_key"])
        timestamp= str(request_config["timestamp_key"])
        path_to_log = str(request_config['path_to_log'])
        prediction_file_name= str(request_config['prediction_file_name'])

        preprocessor = preprocessing.Preprocessing()

        #: import the partial trace
        try: 
            #: formatting avoided, just a quick import. we assume the input is a csv with "," as separator
            preprocessor.handle_import(False, path_to_log, case_id, timestamp, activity,sep=",", formatting=False )
        except Exception as e: 
            #: notify error
            return {"error": str(e)}, 400

        input_df = preprocessor.event_df

        cuda = False

        #: import the RNN model and its config
        path_to_model = str(request_config["path_to_model"])
        config = nn_manager.Config()
        with open (request_config["config"], "r") as f:
            dic = json.load(f)
            dic["time_precision"] = "NS"
        config.load_config(dic)
        neural_manager = nn_manager.NNManagement(config)
        neural_manager.import_nn_model(path_to_model)


        #: create the prediction amanger instance
        pm = prediction_manager.PredictionManager(
            neural_manager.model, 
            case_id, 
            activity,
            timestamp, 
            config
        )
        #: do the predictions
        try: 
            pm.multiple_prediction_dataframe(
                depth, 
                degree, 
                input_df
            )
        except Exception as e: 
            #: notify error
            return {"error": f"{str(e)}"},400 

        #: write the obtaine predictions
        paths = pm.jsonify_paths()
        with open(prediction_file_name, 'w') as multi_predictions:
            json.dump(paths, multi_predictions, indent=2)
        return ok , 200


@routes.route('/single_prediction', methods = ["POST"])
@check_required_paths_factory(['path_to_log', "config", "path_to_model"])
def single_prediction():
    """
    given a partial trace do one prediction.  

    :param case_id: case id column name
    :param activity_key: activity column name
    :param timestamp: timestamp column name
    :param path_to_log: path to the input partial trace. must contain a single case id and columns 
    must have the same names as the the ones used in the log for training. it must be a csv file with "," as separator
    :param path_to_model: path to the RNN model used for making predictions 
    :param config: path to the config file for the model

    :return predicted_time:  predicted next timestamp
    :return predicted_event:  predicted next activity
    :return probability:  probability of the event
    """
    if request.method == 'POST':
        request_config = request.get_json()
        #: extract params
        case_id= str(request_config["case_id"])
        activity= str(request_config["activity_key"])
        timestamp= str(request_config["timestamp_key"])
        path_to_log = str(request_config['path_to_log'])
        preprocessor = preprocessing.Preprocessing()

        #: import partial trace
        try: 
            #: formatting set to false, we just want to do a quick import
            preprocessor.handle_import(False, path_to_log, case_id, timestamp, activity,sep=",", formatting=False )
        except Exception as e: 
            #: notify errors
            return {"error": str(e)}, 400

        input_df = preprocessor.event_df

        cuda = False #: TODO: check if cuda is also used for making predictions.


        #: load the RNN model and config (language encoders, params, ...)
        path_to_model = str(request_config["path_to_model"])
        config = nn_manager.Config()
        with open (request_config["config"], "r") as f:
            dic = json.load(f)
            dic["time_precision"] = "NS"
        config.load_config(dic)
        neural_manager = nn_manager.NNManagement(config)
        neural_manager.import_nn_model(path_to_model)

        #: create the prediction amanger intance 
        pm = prediction_manager.PredictionManager(
            neural_manager.model, 
            case_id, 
            activity,
            timestamp, 
            config
        )

        #: do the prediction
        try:
            time, event, prob = pm.single_prediction_dataframe(input_df)
        except Exception as e: 
            #: notify error
            return {"error": f"{str(e)}"},400 

        #: return the prediction
        return pm.jsonify_single(time, event, prob)

        


@routes.route('/random_search', methods = ["POST"])
@check_booleans_factory(["cuda", "is_xes"])
@check_integers_factory(["seq_len","batch_size", "epochs", "iterations"])
@check_floats_factory(["lr", "split"])
@check_required_paths_factory(['path_to_log'])
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
        

        #: extract params
        is_xes = request_config["is_xes"] 
        path_to_log = str(request_config['path_to_log'])
        case_id= str(request_config["case_id"])
        activity= str(request_config["activity_key"])
        timestamp= str(request_config["timestamp_key"])
        split = float(request_config["split"])
        iterations = int(request_config["iterations"])
        sp = request_config["search_params"]


        #: check the format for the search params. 
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


        #: import the event log for training
        preprocessor = preprocessing.Preprocessing()
        try: 
            preprocessor.handle_import(is_xes, path_to_log, case_id, timestamp, activity)
        except Exception as e: 
            return {
                "error":str(e), 
                "description": "error while importing"
            }, 400

        
        #: split the event log for trianing
        try:
            train, test= preprocessor.split_train_test(split)
        except exceptions.TrainPercentageTooHigh: 
            return {"error": "train percentage must be in range (0,1) and should not yield empty sublogs"}, 400
        except Exception as e: 
            return {
                "error":str(e),
                "description": "error while importing"
            }, 400

        #: prepare nn manager
        neural_manager= nn_manager.NNManagement() 
        neural_manager.config.cuda =  request_config["cuda"]
        neural_manager.config = load_config_from_preprocessor(neural_manager.config, preprocessor) 
        neural_manager.load_data(train, test, preprocessor.case_id_key, preprocessor.case_timestamp_key, preprocessor.case_activity_key)

        #: do random search 
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
        

        #: export the model and its config 
        config = neural_manager.config.asdict()
        neural_manager.export_nn_model(request_config['model_path'])
        with open(f"{request_config['model_path'][:-3]}.config.json", "w") as f:
            json.dump(config,f)
        data = {
            "acc":  acc
        }


        #: return accuracy
        response = make_response(jsonify(data))




        return response
    




@routes.route('/grid_search', methods = ["POST"])
@check_booleans_factory(["cuda", "is_xes"])
@check_integers_factory(["seq_len","batch_size", "epochs"])
@check_floats_factory(["lr", "split"])
@check_required_paths_factory(['path_to_log'])
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
        
        #: extract params
        is_xes = request_config["is_xes"] 
        path_to_log = str(request_config['path_to_log'])
        case_id= str(request_config["case_id"])
        activity= str(request_config["activity_key"])
        timestamp= str(request_config["timestamp_key"])
        sp = request_config["search_params"]
        split = float(request_config["split"])


      
        #: check the correctness of the search params
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


        #: import training log
        preprocessor = preprocessing.Preprocessing()
        try: 
            preprocessor.handle_import(is_xes, path_to_log, case_id, timestamp, activity)
        except Exception as e: 
            return {
                "error":str(e), 
                "description": "error while importing"
            }, 400

        #: split the training log
        try:
            train, test= preprocessor.split_train_test(split)
        except exceptions.TrainPercentageTooHigh: 
            return {"error": "train percentage must be in range (0,1) and should not yield empty sublogs"}, 400
        except Exception as e: 
            return {
                "error":str(e),
                "description": "error while importing"
            }, 400

        #: initialize nn manager
        neural_manager= nn_manager.NNManagement() 
        neural_manager.config.cuda =  request_config["cuda"]
        neural_manager.config = load_config_from_preprocessor(neural_manager.config, preprocessor) 
        neural_manager.load_data(train, test, preprocessor.case_id_key, preprocessor.case_timestamp_key, preprocessor.case_activity_key)
        
        #: do grid search
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
        

        #: export model and its config
        config = neural_manager.config.asdict()
        neural_manager.export_nn_model(request_config['model_path'])
        with open(f"{request_config['model_path'][:-3]}.config.json", "w") as f:
            json.dump(config,f)
        
        #: return the accuracy
        data = {
            "acc":  acc
        }
        response = make_response(jsonify(data))
        return response

@routes.route('/train_nn', methods = ["POST"])
@check_booleans_factory(["cuda", "is_xes"])
@check_integers_factory(["seq_len", "emb_dim", "hid_dim", "mlp_dim", "batch_size", "epochs"])
@check_floats_factory(["lr", "split"])
@check_required_paths_factory(['path_to_log'])
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

        #: extract params
        request_config = request.get_json()
        path_to_log = str(request_config['path_to_log'])
        case_id= str(request_config["case_id"])
        activity= str(request_config["activity_key"])
        timestamp= str(request_config["timestamp_key"])


        preprocessor = preprocessing.Preprocessing()
        

        #: import the log for training
        try: 
            preprocessor.handle_import(request_config['is_xes'], path_to_log, case_id, timestamp, activity)
        except Exception as e: 
            return {
                "error":str(e), 
                "description": "error while importing"
            }, 400

        #: split the log for training
        try:
            train, test= preprocessor.split_train_test(float(request_config["split"]))
        except exceptions.TrainPercentageTooHigh: 
            return {"error": "train percentage must be in range (0,1) and should not yield empty sublogs"}, 400
        except Exception as e: 
            return {
                "error":str(e),
                "description": "error while importing"
            }, 400


        #: initialize the manager
        neural_manager= nn_manager.NNManagement() 
        neural_manager.config = load_config_from_params(neural_manager.config, request_config)
        neural_manager.config = load_config_from_preprocessor(neural_manager.config, preprocessor) 
        neural_manager.load_data(train, test, case_id, timestamp, activity)

        #: train the RNN
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


        #: jsonify trianing stats for response
        training_stats = neural_manager.get_training_statistics()
        data = {
            "training_statistics": training_stats, 
        }


        #: export model and its config
        config = neural_manager.config.asdict()
        neural_manager.export_nn_model(request_config['model_path'])
        with open(f"{request_config['model_path'][:-3]}.config.json", "w") as f:
            json.dump(config,f)

        #: respond with train stats
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
@check_required_paths_factory(['path_to_log'])
@check_not_present_paths_factory(["save_path"])
def replace_with_mode():
    """
    replaces NaN's in the activity column with median

    a filtered event log is created in the given path

    the following data is expected in the JSON body of the request: 
    :param path_to_log: path to the log used for training. must not be encoded
    :param case_id: name of case id column
    :param activity_key: name of activity column
    :param timestamp_key: name of timestamp column
    :param save_path: path where the processed event log is exported 
    """
    if request.method == 'POST':
        request_config = request.get_json()
        #: extract params
        is_xes = request_config["is_xes"] 
        path_to_log = str(request_config['path_to_log'])
        case_id= str(request_config["case_id"])
        activity= str(request_config["activity_key"])
        timestamp= str(request_config["timestamp_key"])
        save_path= str(request_config["save_path"])
        if not is_xes: 
            sep= str(request_config["sep"])
        else: 
            sep = ""


        preprocessor = preprocessing.Preprocessing()
        #: import the log
        try: 
            preprocessor.handle_import(is_xes, path_to_log, case_id, timestamp, activity,sep=sep, formatting=False )
        except Exception as e: 
            return {
                "error":str(e),
                "description": "error while importing"
            }, 400

        success= preprocessor.replace_activity_nan_with_mode()
        if success:
            #: notify success
            preprocessor.event_df.to_csv(save_path, sep = ",")
            return {
                "status": "successfully finished", 
                "save_path":save_path
                }, 200
        
        #: notify error
        return {"error": "nan replacement went wrong..."}, 400





@routes.route('/add_unique_start_end', methods = ["POST"])
@check_booleans_factory(["is_xes"])
@check_required_paths_factory(['path_to_log'])
@check_not_present_paths_factory(["save_path"])
def add_unique_start_end():
    """
    adds a unique start/end activity to the log

    a filtered event log is created in the given path

    the following data is expected in the JSON body of the request: 
    :param path_to_log: path to the log used for training. must not be encoded
    :param case_id: name of case id column
    :param activity_key: name of activity column
    :param timestamp_key: name of timestamp column
    :param save_path: path where the processed event log is exported 
    """
    if request.method == 'POST':
        request_config = request.get_json()

        #: extract params
        is_xes = request_config["is_xes"] 
        path_to_log = str(request_config['path_to_log'])
        case_id= str(request_config["case_id"])
        activity= str(request_config["activity_key"])
        timestamp= str(request_config["timestamp_key"])
        save_path= str(request_config["save_path"])

        if not is_xes: 
            sep= str(request_config["sep"])
        else: 
            sep = "" 

        preprocessor = preprocessing.Preprocessing()

        #: import log
        try: 
            preprocessor.handle_import(is_xes, path_to_log, case_id, timestamp, activity,sep=sep, formatting=False )
        except Exception as e: 
            return {
                "error":str(e),
                "description": "error while importing"
            }, 400


        success= preprocessor.add_unique_start_end_activity()
        if success:
            #: notify success
            preprocessor.event_df.to_csv(save_path, sep = ",")
            return {
                "status": "successfully created", 
                "save_path":save_path
                }, 200
        #: notify error
        return {"error": "operation not necessary, the log already has a unique start/end activity"}, 400

@routes.route('/remove_duplicates', methods = ["POST"])
@check_booleans_factory(["is_xes"])
@check_required_paths_factory(['path_to_log'])
@check_not_present_paths_factory(["save_path"])
def remove_duplicates():
    """
    removes the duplicates ie the rows where the same activity happened at the same time in the same case id.

    a filtered event log is created in the given path

    the following data is expected in the JSON body of the request: 
    :param path_to_log: path to the log used for training. must not be encoded
    :param case_id: name of case id column
    :param activity_key: name of activity column
    :param timestamp_key: name of timestamp column
    :param save_path: path where the processed event log is exported 
    """
    if request.method == 'POST':
        request_config = request.get_json()

        #: extract params
        is_xes = request_config["is_xes"] 


        path_to_log = str(request_config['path_to_log'])
        case_id= str(request_config["case_id"])
        activity= str(request_config["activity_key"])
        timestamp= str(request_config["timestamp_key"])
        save_path= str(request_config["save_path"])

        if not is_xes: 
            sep= str(request_config["sep"])
        else:
            sep = ""

        preprocessor = preprocessing.Preprocessing()
        #: import the log
        try: 
            preprocessor.handle_import(is_xes, path_to_log, case_id, timestamp, activity,sep=sep, formatting=False )
        except Exception as e: 
            return {
                "error":str(e),
                "description": "error while importing"
            }, 400


        success= preprocessor.remove_duplicate_rows()
        if success:
            #: notify success
            path = save_path
            preprocessor.event_df.to_csv(save_path, sep = ",")
            return {"save_path":save_path}, 200

        #: notify error
        return {"error": "removing duplicates went wrong..."},400 





