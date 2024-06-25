"""
This module contains all the supported server routes by ProcessProphet. 

A flask server is implemented that runs on port `8080` by default. This can be changed
in the `.env` file. 

The server has been designed assuming both frontend and this server share the same 
file system, as the server writes output files directly to the indicated directories, instead of
returning them as a response.
"""


from flask import Blueprint, request, make_response, jsonify, Response
from server import preprocessing
from server import nn_manager
from server import process_model_manager
from server import prediction_manager
from server import exceptions
from functools import wraps
from server import loggers
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
    Server route: `/conformance`

    Applies a conformance checking algorithm on the given `petri_net_path` and the log in `path_to_log`. Currently only
    token-based replay and alignment based conformance checking are supported. The conformance checking technique is selected by the `conformance_technique` parameter.

    The POST request must have the following parameters:

    Args:
        is_xes (bool): Whether the input log is in XES format or not (otherwise CSV).
        case_id (str): Case ID column name.
        activity_key (str): Activity column name.
        timestamp (str): Timestamp column name.
        path_to_log (str): Path to the event log.
        petri_net_path (str): Path to the Petri net used for conformance checking.
        conformance_technique (str): Either `"token"` or `"alignment"`. This selects the corresponding conformance checking technique.

    400 Response:
        - An object with the "error" key indicating what went wrong is sent.

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
    Server route: `/generate_predictive_process_model`

    Create a predictive process model, i.e., a petri net using the predictive log in `path_to_log` and the given configuration.
    The petri net is generated using process mining algorithms such as the alpha miner, heuristic miner, inductive miner, and prefix tree miner, 
    which can be selected using the `selected_model` parameter. The petri net is saved in the `petri_net_path` and the config file is saved in the `petri_net_path.json`.


    The POST request must have the following parameters:

    Args:
        is_xes (bool): Whether the input log is in XES format or not (otherwise CSV).
        case_id (str): The column name for the case ID.
        activity_key (str): The column name for the activity.
        timestamp (str): The column name for the timestamp.
        path_to_log (str): The path to the event log.
        petri_net_path (str): The path where the PNML file and the JSON file should be exported.
        selected_model (str): The selected mining model ("alpha_miner", "heuristic_miner", "inductive_miner", "prefix_tree_miner").
        mining_algo_config (dict): The settings for the selected process mining algorithm.
        sep (str): The column separator (used for CSV files).
        config (str): The path to the config file for the model.


    200 response side effects:
        - The petri net is saved in the petri_net_path. 
        - The petri net config is saved in the petri_net_path.json.

    400 Response:
        - An object with the "error" key indicating what went wrong is sent.
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
    Server route: `/generate_predictive_log`

    Generates the predictive event log by cutting all traces using the given configuration
    and by extending these cut traces with predictions. The predictive log is exported to `new_log_path`.
    The cutting can be done in two ways: either by cutting the last `cut_length` events from each trace or by cutting at a random sequence index.
    If cutting at random indices, predictions can be made until an end marking is reached (`non_stop==True`) or for a fixed number of iterations (`non_stop= False`).
    If `non_stop==False`, or `random_cuts==False`, each trace is extended by `cut_length` predictions.
    A pytorch model found in `path_to_model` is used for making the predictions.

    The POST request must have the following parameters:

    Args:
        is_xes (bool): Whether the input log is xes or not (otherwise csv). 
        case_id (str): Case id column name.
        activity_key (str): Activity column name.
        timestamp (str): Timestamp column name.
        path_to_log (str): Path to the event log used for cutting.
        path_to_model (str): Path to the RNN model used for making predictions.
        new_log_path (str): Path where the predictive log should be saved (csv format is default). 
        sep (str): Column separator (used for csv input logs).
        config (str): Path to the config file for the model.
        random_cuts (bool): If set to True, each trace is cut at a random sequence index. 
        non_stop (bool): If set to True, predictions are made until an end marking is reached. 
        cut_length (int): In case of random cuts = non_stop = False, we cut from the tail of each trace 
            the last `cut_length` events. 
        upper (int): Upper bound for the number of iterations the non_stop variant should run (just for safety).
    
    200 response side effects:
        - The predictive log is saved in the new log path.

    400 Response:
        - An object with the "error" key indicating what went wrong is sent.
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
    Server route: `/multiple_prediction`

    A model is used for making multiple predictions. The predictions are saved in the `prediction_file_name` file.
    A tree like structure is generated with the predictions. The tree has a depth of `depth` and a branching degree of `degree`.

    The POST request must have the following parameters:

    Args:
        case_id (str): Case ID column name
        activity_key (str): Activity column name
        timestamp (str): Timestamp column name
        path_to_log (str): Path to the input partial trace. Must contain a single case ID and columns with the same names as the ones used in the log for training. It must be a CSV file with "," as the separator.
        path_to_model (str): Path to the RNN model used for making predictions
        prediction_file_name (str): File name for the output file that will contain the predictions
        config (str): Path to the config file for the model
        degree (int): Branching degree of the generated prediction tree
        depth (int): Depth that the predictive tree should have


    200 response side effects:
        - The predictions are saved in the prediction file in the path `prediction_file_name`.
        The generated object contains a "paths" key, which is a list of objects. 
        Each object has a list of pairs (the sequence) and a probability.

    400 Response:
        - An object with the "error" key indicating what went wrong is sent.
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

     





        #: import the RNN model and its config
        path_to_model = str(request_config["path_to_model"])
        config = nn_manager.Config()
        with open (request_config["config"], "r") as f:
            dic = json.load(f)
            dic["time_precision"] = "NS"
        
        config.load_config(dic)


        # ;import partial trace
        preprocessor2 = preprocessing.Preprocessing()
        try: 
            #: do this to avoid recompiting the language encoders
            preprocessor2.activity_le = config.activity_le
            preprocessor2.case_id_le= config.case_id_le
            preprocessor2.handle_import(False, path_to_log, case_id, timestamp, activity,sep=",", formatting=True)

        except Exception as e: 
            #: notify error
            return {"error": str(e)}, 400



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
                preprocessor2.event_df
            )
            
            pm.encoded_df = preprocessor2.event_df
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
    Server route: `/single_prediction`

    Given a partial trace found in `path_to_log`, perform a single prediction.

    The POST request must have the following parameters:

    Args:
        case_id (str): Case ID column name.
        activity_key (str): Activity column name.
        timestamp (str): Timestamp column name.
        path_to_log (str): Path to the input partial trace. Must contain a single case ID and columns 
                           with the same names as the ones used in the log for training. 
                           It must be a CSV file with "," as the separator.
        path_to_model (str): Path to the RNN model used for making predictions.
        config (str): Path to the config file for the model.

    Returns:
        predicted_time (float): Predicted next timestamp.
        predicted_event (str): Predicted next activity.
        probability (float): Probability of the event.


    400 Response:
        - An object with the "error" key indicating what went wrong is sent.
    """
    if request.method == 'POST':
        request_config = request.get_json()
        #: extract params
        case_id= str(request_config["case_id"])
        activity= str(request_config["activity_key"])
        timestamp= str(request_config["timestamp_key"])
        path_to_log = str(request_config['path_to_log'])
       


        #: load the RNN model and config (language encoders, params, ...)
        path_to_model = str(request_config["path_to_model"])
        config = nn_manager.Config()
        with open (request_config["config"], "r") as f:
            dic = json.load(f)
            dic["time_precision"] = "NS"
        config.load_config(dic)


        #: import partial trace
        preprocessor = preprocessing.Preprocessing()
        try: 
            #: do this to avoid recomputing the language encoders
            preprocessor.activity_le = config.activity_le
            preprocessor.case_id_le= config.case_id_le
            preprocessor.handle_import(False, path_to_log, case_id, timestamp, activity,sep=",", formatting=True)

        except Exception as e: 
            #: notify error
            return {"error": str(e)}, 400




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
            time, event, prob = pm.single_prediction_dataframe(
                preprocessor.event_df
            )
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
    Server route: `/random_search`

    Apply random search on the given log in `path_to_log` for training and testing. 
    The best model is saved in `model_path`. The parameters are listed below.

    The POST request must have the following parameters:

    Args:
        path_to_log (str): Path to the log used for training. Must not be encoded.
        model_path (str): Path where the model should be saved.
        split (float): Float in the range [0, 1]. Represents train-test ratio. 
        case_id (str): Name of the case ID column.
        activity_key (str): Name of the activity column.
        timestamp_key (str): Name of the timestamp column.
        cuda (bool): True/False if CUDA is used or not. 
        seq_len (int): Length of the sliding window used. 
        lr (float): Learning rate.
        batch_size (int): Batch size. 
        epochs (int): Number of epochs.
        is_xes (bool): Is the log in XES format?
        iterations (int): Number of iterations for random search.
        search_params (dict): Dictionary of the format: 
            {
                "hid_dim": [lower_bound (int), upper_bound (int)],
                "mlp_dim": [lower_bound (int), upper_bound (int)],
                "emb_dim": [lower_bound (int), upper_bound (int)]
            }

    Returns:
        config (dict): The config file that is used for Process Prophet. 
        acc (float): The best accuracy achieved during training.
        model (str): A base64 encoded PT file containing the model setup ready for importing.

    Raises:
        ValueError: If any of the input parameters are invalid.

    200 response side effects:
        - The config file used for Process Prophet is saved in the model path with the extension `.config.json`.
        - The trained model is saved in the model path. 

    400 Response:
        - An object with the "error" key indicating what went wrong is sent.
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
    Server route: `/grid_search`

    Apply grid search on the given log in `path_to_log` for training and testing. The best model is saved in `model_path`.

    The POST request must have the following parameters:

    Args:
        path_to_log (str): Path to the log used for training. Must not be encoded.
        model_path (str): Path where the model should be saved.
        split (float): Float in the range [0, 1] representing the train-test ratio. 
        case_id (str): Name of the case ID column.
        activity_key (str): Name of the activity column.
        timestamp_key (str): Name of the timestamp column.
        cuda (bool): True/False indicating whether CUDA is used or not. 
        seq_len (int): Length of the sliding window used. 
        lr (float): Learning rate.
        batch_size (int): Batch size. 
        epochs (int): Number of epochs.
        is_xes (bool): Is the log in XES format?
        search_params (dict): Dictionary of the format: 
            {
                "hid_dim": [lower_bound, upper_bound, step],
                "mlp_dim": [lower_bound, upper_bound, step],
                "emb_dim": [lower_bound, upper_bound, step]
            }

    Returns:
        dict: The response contains the following and has the following side effects: 
            - `config`: The config file used for Process Prophet. 
            - `acc`: The best accuracy achieved during training.
            - `model`: A base64 encoded PT file containing the model setup ready for importing.

    200 response side effects:
        - The config file used for Process Prophet is saved in the model path with the extension `.config.json`.
        - The trained model is saved in the model path. 
    
    400 Response:
        - An object with the "error" key indicating what went wrong is sent.
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
    Server route: `/train_nn`

    Trains the RMTPP neural network using the log in `path_to_log` for training and testing. 
    A model is generated in `model_path` and the config file is saved in `model_path` with the extension `.config.json`.
    All trainig params are listed below. 

    The POST request must have the following parameters:

    Args:
        path_to_log (str): Path to the log used for training. Must not be encoded.
        model_path (str): Path where the model should be saved.
        split (float): Float in the range [0,1] representing the train-test ratio.
        case_id (str): Name of the case id column.
        activity_key (str): Name of the activity column.
        timestamp_key (str): Name of the timestamp column.
        cuda (bool): True/False indicating whether CUDA is used or not.
        seq_len (int): Length of the sliding window used. Also affects tensor dimension.
        lr (float): Learning rate.
        batch_size (int): Batch size.
        epochs (int): Number of epochs.
        is_xes (bool): Is the log in XES format?
        emb_dim (int): Embedding dimension.
        hid_dim (int): Hidden layer dimension.
        mlp_dim (int): MLP dimension.

    200 response side effects:
        - The config file used for Process Prophet is saved in the model path with the extension `.config.json`.
        - The trained model is saved in the model path. 

    400 Response:
        - An object with the "error" key indicating what went wrong is sent.
    
    Returns:
        acc: The training accuracy achieved.
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
    Server route: `/replace_with_mode`

    Replaces NaN's in the activity column with the median to the event log in in `path_to_log`.
    Creates a filtered event log in `save_path`.

    The POST request must have the following parameters:

    Args:
        path_to_log (str): Path to the log used for training. Must not be encoded.
        save_path (str): Path where the processed event log is exported.
        case_id (str): Name of the case id column.
        activity_key (str): Name of the activity column.
        timestamp_key (str): Name of the timestamp column.

        
    200 Response sideffects: 
        - The filtered event log is saved in `save_path`.


    400 Response:
        - An object with the "error" key indicating what went wrong is sent.
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
    Server route: `/replace_unique_start_end`

    Adds a unique start/end activity to the log in `path_to_log`.

    A filtered event log is created in `save_path`.

    The POST request must have the following parameters:

    Args:
        path_to_log (str): Path to the log used for training. Must not be encoded.
        save_path (str): Path where the processed event log is exported.
        case_id (str): Name of the case ID column.
        activity_key (str): Name of the activity column.
        timestamp_key (str): Name of the timestamp column.

    200 Response sideffects: 
        - The filtered event log is saved in `save_path`.

    400 Response:
        - An object with the "error" key indicating what went wrong is sent.
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
    Server route: `/remove_duplicates`

    Removes the duplicates from the event log in `path_to_log`.

    This function removes the rows where the same activity happened at the same time in the same case ID.
    A filtered event log is created in `save_path`.

    The POST request must have the following parameters:

    Args:
        path_to_log (str): Path to the log used for training. Must not be encoded.
        save_path (str): Path where the processed event log is exported.
        case_id (str): Name of the case ID column.
        activity_key (str): Name of the activity column.
        timestamp_key (str): Name of the timestamp column.

    Returns:
        dict: A dictionary containing the save path of the processed event log.

    200 Response sideffects: 
        - The filtered event log is saved in `save_path`.
        
    400 Response:
        - An object with the "error" key indicating what went wrong is sent.
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





