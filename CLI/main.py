import requests
import json
import os
import pprint
from pytermgui import tim, Container, Label, Splitter, Button, Checkbox
import pytermgui as ptg
import time
from ProcessProphetStart import ProcessProphetStart
from ProcessProphet import ProcessProphet
from dotenv import load_dotenv



load_dotenv()
SERVER_NAME= os.getenv('SERVER_NAME')
SERVER_PORT= os.getenv('SERVER_PORT')




def test_train():
    models_base_path = "CLI/models/"
    model_name = "my_cool_model.pt"
    params = {
        "is_xes": False, 
        "path_to_log": "data/train_day_joined.csv", 
        "sep": ",", 
        "case_id": "case_id", 
        "activity_key": "activity", 
        "timestamp_key": "timestamp", 
        "cuda": True, 
        "split": 0.9, 
        "model_name": model_name
    }
    response = requests.get(
        f"http://{SERVER_NAME}:{SERVER_PORT}/train_nn", 
        params = params,
        timeout =6000
    )
    print(response.status_code)
    #print(response.headers)
    metadata = json.loads(response.headers.get("X-Metadata"))
    statistics = metadata["training_statistics"]
    config = metadata["config"]
    with open(models_base_path+ model_name, 'wb') as file:
        # Write the content of the response to the file
        file.write(response.content) 
    print("done saving")

    return statistics, config


def test_predictive_process_model_generator(
        config,
        path_to_model, 
        path_to_log, 
        new_log_path, 
        selected_model, 
        petri_net_path, 
        mining_algo_config
    ):

    params = {
        "is_xes": False, 
        "path_to_log": path_to_log, 
        "sep": ",", 
        "case_id": "case_id", 
        "activity_key": "activity", 
        "timestamp_key": "timestamp", 
        "cuda": True, 
        "path_to_model": path_to_model, 
        "config": json.dumps(config), 
        "new_log_path": new_log_path, 
        "selected_model":selected_model, 
        "petri_net_path":petri_net_path, 
        "mining_algo_config": json.dumps(mining_algo_config)
    }


    response = requests.get(
        f"http://{SERVER_NAME}:{SERVER_PORT}/generate_predictive_process_model", 
        params = params,
        timeout =6000
    )
    return json.loads(response.text)




def test_predictive_log_generator(
        config,
        path_to_model, 
        path_to_log, 
        non_stop, 
        upper, 
        random_cuts, 
        cut_length, 
        new_log_path
    ):
    params = {
        "is_xes": False, 
        "path_to_log": path_to_log, 
        "sep": ",", 
        "case_id": "case_id", 
        "activity_key": "activity", 
        "timestamp_key": "timestamp", 
        "cuda": True, 
        "path_to_model": path_to_model, 
        "config": json.dumps(config), 
        "non_stop": non_stop, 
        "upper": upper, 
        "random_cuts": random_cuts, 
        "cut_length": cut_length, 
        "new_log_path": new_log_path
    }


    response = requests.get(
        f"http://{SERVER_NAME}:{SERVER_PORT}/generate_predictive_log", 
        params = params,
        timeout =6000
    )
    return json.loads(response.text)

def test_multiple_prediction(config, path_to_model, path_to_log, depth, degree):
    params = {
        "is_xes": False, 
        "path_to_log": path_to_log, 
        "sep": ",", 
        "case_id": "case_id", 
        "activity_key": "activity", 
        "timestamp_key": "timestamp", 
        "cuda": True, 
        "split": 0.9, 
        "path_to_model": path_to_model, 
        "config": json.dumps(config), 
        "depth": depth,
        "degree":degree 
    }


    response = requests.get(
        f"http://{SERVER_NAME}:{SERVER_PORT}/multiple_prediction", 
        params = params,
        timeout =6000
    )
    print(response.status_code)
    return json.loads(response.text)


def test_single_prediction(config, path_to_model, path_to_log):
    params = {
        "is_xes": False, 
        "path_to_log": path_to_log, 
        "sep": ",", 
        "case_id": "case_id", 
        "activity_key": "activity", 
        "timestamp_key": "timestamp", 
        "cuda": True, 
        "split": 0.9, 
        "path_to_model": path_to_model, 
        "config": json.dumps(config), 
    }


    response = requests.get(
        f"http://{SERVER_NAME}:{SERVER_PORT}/single_prediction", 
        params = params,
        timeout =6000
    )
    return json.loads(response.text)





if __name__=="__main__":
    #statistics, config = test_train()
    #test_single_prediction(config, "CLI/models/my_cool_model.pt", "CLI/input_logs/dummy.csv")
    #print(test_multiple_prediction(config, "CLI/models/my_cool_model.pt", "CLI/input_logs/dummy.csv", 2,2))
    """
    test_predictive_log_generator(
        config, 
        "CLI/models/my_cool_model.pt", 
        "data/train_day_joined.csv", 
        True, 
        30, 
        True, 
        3, 
        "CLI/predictive_logs/my_cool_predictive_log.csv"
    )
    test_predictive_process_model_generator(
        config, 
        "CLI/models/my_cool_model.pt", 
        "data/train_day_joined.csv", 
        "CLI/predictive_logs/my_cool_predictive_log.csv", 
        "alpha_miner", 
        "CLI/petri_nets/alpha.pnml", 
        {}
    )
    """
    #start_process_prophet()
    pp = ProcessProphet()
    pps = ProcessProphetStart(pp)
    pp.run()