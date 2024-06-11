from server import preprocessing
from server import nn_manager
from server import server_routes 
from server import loggers
from server import prediction_manager
from server import  process_model_manager 
from server import time_precision


from flask import Flask
import time
import logging
#from ray import tune
import random
import os
from dotenv import load_dotenv
import torch
import json
import pm4py


#: load server port form env variable. 
load_dotenv()
SERVER_PORT= os.getenv('SERVER_PORT')






#: initialize flask app and load the routes from the `server_routes` file
app = Flask(__name__)
app.register_blueprint(server_routes.routes)



def HL_shorter():
    #: this function is used for debugging PLEASE IGNORE



    preprocessor = preprocessing.Preprocessing()
    #path =  "data/train_day_joined.csv"
    #path = "data/BPI_Challenge_2019.xes"
    path = "data/Hospital_log.xes"
    #path = "data/dummy.csv"
    #path =  "data/running.csv"

    preprocessor.handle_import(True,path,"case:concept:name", "time:timestamp", "concept:name") 
    df = preprocessor.unencoded_df[:1000]
    pm4py.write_xes(df, "Hospital_log_mini.xes")
    pm4py.write_xes(df, "Hospital_log_mini1.xes")




def generate_hospital_mini():
    #: this function is used for debugging PLEASE IGNORE

    
    preprocessor = preprocessing.Preprocessing()
    #path =  "data/train_day_joined.csv"
    #path = "data/BPI_Challenge_2019.xes"
    path = "data/Hospital_log.xes"
    #path = "data/dummy.csv"
    #path =  "data/running.csv"

    preprocessor.handle_import(True,path,"case:concept:name", "time:timestamp", "concept:name") 
    df = preprocessor.unencoded_df
    df = df[:20]
    df.to_csv("partial_input.csv")





def dummy():
    #: this function is used for debugging PLEASE IGNORE
    """
    preprocessor = preprocessing.Preprocessing()
    is_xes = False
    #path =  "data/train_day_joined.csv"
    #path = "data/BPI_Challenge_2019.xes"
    path = "data/Hospital_log.xes"
    #path = "data/dummy.csv"
    #path =  "data/running.csv"
    preprocessor.handle_import()
    if is_xes:
        #preprocessor.import_event_log_xes(path , "case:concept:name", "concept:name", "time:timestamp")# hospital
        preprocessor.import_event_log_xes(path , "case:concept:name", "concept:name", "time:timestamp")# bpi 2019
        print(preprocessor.event_df.head())
    else:
        preprocessor.import_event_log_csv(path , "case_id", "activity", "timestamp", ',')

    pm = prediction_manager.PredictionManager(None, "case_id", "activity", "timestamp", None)

    dummy  = pm.get_dummy_process(preprocessor.event_df,"case_id" )
    dummy.to_csv("CLI/input_logs/dummy.csv",sep = ',' )
    """

    preprocessor = preprocessing.Preprocessing()
    #path =  "data/train_day_joined.csv"
    #path = "data/BPI_Challenge_2019.xes"
    path = "data/Hospital_log.xes"
    #path = "data/dummy.csv"
    #path =  "data/running.csv"

    preprocessor.handle_import(True,path,"case:concept:name", "time:timestamp", "concept:name") 
    df = preprocessor.unencoded_df
    df = df[:20]
    df.to_csv("partial_input.csv")


def test_end_activities():
    #: this function is used for debugging PLEASE IGNORE
    preprocessor = preprocessing.Preprocessing()
    is_xes = True 
    #path =  "data/train_day_joined.csv"
    #path = "data/BPI_Challenge_2019.xes"
    path = "data/Hospital_log.xes"
    #path = "data/dummy.csv"
    #path =  "data/running.csv"
    if is_xes:
        #preprocessor.import_event_log_xes(path , "case:concept:name", "concept:name", "time:timestamp")# hospital
        preprocessor.import_event_log_xes(path , "case:concept:name", "concept:name", "time:timestamp")# bpi 2019
        print(preprocessor.event_df.head())
    else:
        preprocessor.import_event_log_csv(path , "case_id", "activity", "timestamp", ',')
    print(preprocessor.find_end_activities())

 
def test_our():
    #: this function is used for debugging PLEASE IGNORE
    preprocessor = preprocessing.Preprocessing()
    #path = "data/BPI_Challenge_2019.xes"
    path = "data/Hospital_log.xes"
    #path = "data/dummy.csv"
    #path =  "data/running.csv"

    #preprocessor.handle_import(False,"data/train_day_joined.csv",  "case_id",  "timestamp","activity",time_precision.TimePrecision.S, ',' )
    preprocessor.handle_import(True,path, "case:concept:name", "time:timestamp","concept:name" ,time_precision.TimePrecision.NS)

    
    train, test = preprocessor.split_train_test(.7)
    neural_manager = nn_manager.NNManagement()

    # select cuda or not
    neural_manager.config.cuda = True

    neural_manager.config.absolute_frequency_distribution = preprocessor.absolute_frequency_distribution
    neural_manager.config.number_classes = preprocessor.number_classes
    neural_manager.load_data(train, test, preprocessor.case_id_key, preprocessor.case_timestamp_key, preprocessor.case_activity_key)
    neural_manager.train()
    stats_in_json = neural_manager.get_training_statistics()
    neural_manager.export_nn_model()

def test_grid_search():
    #: this function is used for debugging PLEASE IGNORE

    preprocessor = preprocessing.Preprocessing()
    is_xes  =False
    path =  "data/train_day_joined.csv"
    #path = "data/BPI_Challenge_2019.xes"
    #path = "data/Hospital_log.xes"
    #path = "data/dummy.csv"
    #path =  "data/running.csv"
     
    if is_xes:
        #preprocessor.import_event_log_xes(path , "case:concept:name", "concept:name", "time:timestamp")# hospital
        preprocessor.import_event_log_xes(path , "case:concept:name", "concept:name", "time:timestamp")# bpi 2019
    else:
        preprocessor.import_event_log_csv(path , "case_id", "activity", "timestamp", ',')
    train, test = preprocessor.split_train_test(.9)


    # Define the hyperparameter search space [lower_bound, upper_bound, step_size]
    sp= {
        "hidden_dim": [500, 2000, 500],
        "mlp_dim": [500, 2000,  500],
        "emb_dim": [500, 2000, 500]
    }
    neural_manager = nn_manager.NNManagement()
    neural_manager.config.absolute_frequency_distribution = preprocessor.absolute_frequency_distribution
    neural_manager.config.cuda = True 
    neural_manager.config.number_classes = preprocessor.number_classes
    neural_manager.grid_search(train, test, sp, preprocessor.case_id_key, preprocessor.case_timestamp_key, preprocessor.case_activity_key)
    


    
def test_random_search(iterations):
    #: this function is used for debugging PLEASE IGNORE
    preprocessor = preprocessing.Preprocessing()
    is_xes  =False
    path =  "data/train_day_joined.csv"
    #path = "data/BPI_Challenge_2019.xes"
    #path = "data/Hospital_log.xes"
    #path = "data/dummy.csv"
    #path =  "data/running.csv"
     
    if is_xes:
        #preprocessor.import_event_log_xes(path , "case:concept:name", "concept:name", "time:timestamp")# hospital
        preprocessor.import_event_log_xes(path , "case:concept:name", "concept:name", "time:timestamp")# bpi 2019
        print(preprocessor.event_df.head())
    else:
        preprocessor.import_event_log_csv(path , "case_id", "activity", "timestamp", ',')
    train, test = preprocessor.split_train_test(.9)

    #stats_in_json = nn_manager.get_training_statistics()
    # Define the hyperparameter search space
    sp= {
        "hidden_dim": [500, 2000],
        "lstm_dim": [500, 2000],
        "emb_dim": [500, 2000]
    }
    neural_manager = nn_manager.NNManagement()
    neural_manager.config.absolute_frequency_distribution = preprocessor.absolute_frequency_distribution
    neural_manager.config.cuda = True 
    neural_manager.config.number_classes = preprocessor.number_classes
    neural_manager.random_search(train, test, sp, iterations, preprocessor.case_id_key, preprocessor.case_timestamp_key, preprocessor.case_activity_key)




def test_single_prediction():
    #: this function is used for debugging PLEASE IGNORE
    preprocessor = preprocessing.Preprocessing()
    is_xes  =False
    path =  "data/train_day_joined.csv"
    #path = "data/BPI_Challenge_2019.xes"
    #path = "data/Hospital_log.xes"
    #path = "data/dummy.csv"
    #path =  "data/running.csv"
     
    if is_xes:
        #preprocessor.import_event_log_xes(path , "case:concept:name", "concept:name", "time:timestamp")# hospital
        preprocessor.import_event_log_xes(path , "case:concept:name", "concept:name", "time:timestamp")# bpi 2019
    else:
        preprocessor.import_event_log_csv(path , "case_id", "activity", "timestamp", ',')
    train, test = preprocessor.split_train_test(.9)

    neural_manager = nn_manager.NNManagement()
    # select cuda or not
    neural_manager.config.cuda = True 
    neural_manager.config.absolute_frequency_distribution = preprocessor.absolute_frequency_distribution
    neural_manager.config.number_classes = preprocessor.number_classes
    neural_manager.load_data(train, test, preprocessor.case_id_key, preprocessor.case_timestamp_key, preprocessor.case_activity_key)
    neural_manager.train()

    pm = prediction_manager.PredictionManager(
        neural_manager.model, 
        preprocessor.case_id_key, 
        preprocessor.case_activity_key, 
        preprocessor.case_timestamp_key, 
        neural_manager.config
    )
    dummy = pm.get_dummy_process(preprocessor.event_df, preprocessor.case_id_key)
    print(pm.single_prediction_dataframe(dummy))


  


def test_multiple_prediction():
    #: this function is used for debugging PLEASE IGNORE
    preprocessor = preprocessing.Preprocessing()
    is_xes  = True
    #path =  "data/train_day_joined.csv"
    path = "data/BPI_Challenge_2019.xes"
    #path = "data/Hospital_log.xes"
    #path = "data/dummy.csv"
    #path =  "data/running.csv"
     
    if is_xes:
        #preprocessor.import_event_log_xes(path , "case:concept:name", "concept:name", "time:timestamp")# hospital
        preprocessor.import_event_log_xes(path , "case:concept:name", "concept:name", "time:timestamp")# bpi 2019
    else:
        preprocessor.import_event_log_csv(path , "case_id", "activity", "timestamp", ',')
    train, test = preprocessor.split_train_test(.9)

    neural_manager = nn_manager.NNManagement()
    # select cuda or not
    neural_manager.config.cuda = True 
    neural_manager.config.absolute_frequency_distribution = preprocessor.absolute_frequency_distribution
    neural_manager.config.activity_le = preprocessor.activity_le
    neural_manager.config.case_id_le = preprocessor.case_id_le
    neural_manager.config.number_classes = preprocessor.number_classes
    neural_manager.config.exponent = preprocessor.exponent
    neural_manager.load_data(train, test, preprocessor.case_id_key, preprocessor.case_timestamp_key, preprocessor.case_activity_key)
    neural_manager.train()

    pm = prediction_manager.PredictionManager( neural_manager.model, preprocessor.case_id_key, preprocessor.case_activity_key, preprocessor.case_timestamp_key, neural_manager.config)
    dummy = pm.get_dummy_process(preprocessor.event_df, preprocessor.case_id_key)
    pm.multiple_prediction_dataframe(
        3, 
        2, 
        dummy  
    )

def test_process_model_manager_random_cut_nontstop():
    #: this function is used for debugging PLEASE IGNORE
    preprocessor = preprocessing.Preprocessing()
    is_xes  = False
    path =  "data/train_day_joined.csv"
    #path = "data/BPI_Challenge_2019.xes"
    #path = "data/Hospital_log.xes"
    #path = "data/dummy.csv"
    #path =  "data/running.csv"
     
    if is_xes:
        #preprocessor.import_event_log_xes(path , "case:concept:name", "concept:name", "time:timestamp")# hospital
        preprocessor.import_event_log_xes(path , "case:concept:name", "concept:name", "time:timestamp")# bpi 2019
    else:
        preprocessor.import_event_log_csv(path , "case_id", "activity", "timestamp", ',')
    train, test = preprocessor.split_train_test(.9)

    neural_manager = nn_manager.NNManagement()
    # select cuda or not
    neural_manager.config.cuda = True 
    neural_manager.config.absolute_frequency_distribution = preprocessor.absolute_frequency_distribution
    neural_manager.config.number_classes = preprocessor.number_classes
    neural_manager.load_data(train, test, preprocessor.case_id_key, preprocessor.case_timestamp_key, preprocessor.case_activity_key)
    neural_manager.train()
    neural_manager.config.activity_le = preprocessor.activity_le
    neural_manager.config.case_id_le = preprocessor.case_id_le
    neural_manager.config.exponent = preprocessor.exponent

    pmm = process_model_manager.ProcessModelManager(
        preprocessor.event_df, 
        neural_manager.model, 
        neural_manager.config,
        preprocessor.case_activity_key,
        preprocessor.case_id_key,
        preprocessor.case_timestamp_key
    )
    pmm.unencoded_df= preprocessor.unencoded_df
    pmm.end_activities = preprocessor.find_end_activities()
    pmm.generate_predictive_log(non_stop=True, upper =30, random_cuts=True, new_log_path="generated_predicted_df.csv")
    pmm.decode_df(pmm.predictive_df).to_csv("generated_predicted_df.csv")

def test_process_model_manager_random_cut():
    #: this function is used for debugging PLEASE IGNORE
    preprocessor = preprocessing.Preprocessing()
    is_xes  = False
    path =  "data/train_day_joined.csv"
    #path = "data/BPI_Challenge_2019.xes"
    #path = "data/Hospital_log.xes"
    #path = "data/dummy.csv"
    #path =  "data/running.csv"
     
    if is_xes:
        #preprocessor.import_event_log_xes(path , "case:concept:name", "concept:name", "time:timestamp")# hospital
        preprocessor.import_event_log_xes(path , "case:concept:name", "concept:name", "time:timestamp")# bpi 2019
    else:
        preprocessor.import_event_log_csv(path , "case_id", "activity", "timestamp", ',')
    train, test = preprocessor.split_train_test(.9)

    neural_manager = nn_manager.NNManagement()
    # select cuda or not
    neural_manager.config.cuda = True 
    neural_manager.config.absolute_frequency_distribution = preprocessor.absolute_frequency_distribution
    neural_manager.config.number_classes = preprocessor.number_classes
    neural_manager.load_data(train, test, preprocessor.case_id_key, preprocessor.case_timestamp_key, preprocessor.case_activity_key)
    neural_manager.train()
    neural_manager.config.activity_le = preprocessor.activity_le
    neural_manager.config.case_id_le = preprocessor.case_id_le
    neural_manager.config.exponent = preprocessor.exponent
    
    pmm = process_model_manager.ProcessModelManager(
        preprocessor.event_df, 
        neural_manager.model, 
        neural_manager.config,
        preprocessor.case_activity_key,
        preprocessor.case_id_key,
        preprocessor.case_timestamp_key
    )
    pmm.unencoded_df= preprocessor.unencoded_df
    pmm.generate_predictive_log(non_stop=False, upper =100, random_cuts=True, new_log_path = "generated_predicted_df.csv")
    pmm.decode_df(pmm.predictive_df).to_csv("generated_predicted_df.csv")

def test_process_model_manager_tail_cut():
    #: this function is used for debugging PLEASE IGNORE
    preprocessor = preprocessing.Preprocessing()
    is_xes  =False
    path =  "data/train_day_joined.csv"
    #path = "data/BPI_Challenge_2019.xes"
    #path = "data/Hospital_log.xes"
    #path = "data/dummy.csv"
    #path =  "data/running.csv"
     
    if is_xes:
        #preprocessor.import_event_log_xes(path , "case:concept:name", "concept:name", "time:timestamp")# hospital
        preprocessor.import_event_log_xes(path , "case:concept:name", "concept:name", "time:timestamp")# bpi 2019
    else:
        preprocessor.import_event_log_csv(path , "case_id", "activity", "timestamp", ',')
    train, test = preprocessor.split_train_test(.9)

    neural_manager = nn_manager.NNManagement()
    # select cuda or not
    neural_manager.config.cuda = True 
    neural_manager.config.absolute_frequency_distribution = preprocessor.absolute_frequency_distribution
    neural_manager.config.number_classes = preprocessor.number_classes
    neural_manager.load_data(train, test, preprocessor.case_id_key, preprocessor.case_timestamp_key, preprocessor.case_activity_key)
    neural_manager.train()
    neural_manager.config.activity_le = preprocessor.activity_le
    neural_manager.config.case_id_le = preprocessor.case_id_le
    neural_manager.config.exponent = preprocessor.exponent
    

    pmm = process_model_manager.ProcessModelManager(
        preprocessor.event_df, 
        neural_manager.model, 
        neural_manager.config,
        preprocessor.case_activity_key,
        preprocessor.case_id_key,
        preprocessor.case_timestamp_key
    )

    pmm.generate_predictive_log(non_stop=False, random_cuts=False, cut_length = 3)


def test_alpha_miner():

    #: this function is used for debugging PLEASE IGNORE
    preprocessor = preprocessing.Preprocessing()
    is_xes  = True
    path =  "data/train_day_joined.csv"
    #path = "data/BPI_Challenge_2019.xes"
    #path = "data/Hospital_log.xes"
    #path = "data/dummy.csv"
    #path =  "data/running.csv"
     
    if is_xes:
        #preprocessor.import_event_log_xes(path , "case:concept:name", "concept:name", "time:timestamp")# hospital
        preprocessor.import_event_log_xes(path , "case:concept:name", "concept:name", "time:timestamp")# bpi 2019
    else:
        preprocessor.import_event_log_csv(path , "case_id", "activity", "timestamp", ',')
    train, test = preprocessor.split_train_test(.9)

    neural_manager = nn_manager.NNManagement()
    # select cuda or not
    neural_manager.config.cuda = True 
    neural_manager.config.absolute_frequency_distribution = preprocessor.absolute_frequency_distribution
    """
    neural_manager.config.epochs = 15
    neural_manager.config.emb_dim =1000
    neural_manager.config.hid_dim =1000
    neural_manager.config.mlp_dim =1000
    """
    neural_manager.config.number_classes = preprocessor.number_classes
    neural_manager.load_data(train, test, preprocessor.case_id_key, preprocessor.case_timestamp_key, preprocessor.case_activity_key)
    neural_manager.train()
    neural_manager.config.activity_le = preprocessor.activity_le
    neural_manager.config.case_id_le = preprocessor.case_id_le
    neural_manager.config.exponent = preprocessor.exponent
    
    pmm = process_model_manager.ProcessModelManager(
        preprocessor.event_df, 
        neural_manager.model, 
        neural_manager.config,
        preprocessor.case_activity_key,
        preprocessor.case_id_key,
        preprocessor.case_timestamp_key
    )
    pmm.end_activities = preprocessor.find_end_activities()
    #pmm.generate_predictive_log_random_cut_until_end(100)
    pmm.generate_predictive_log_random_cut(100)
    pmm.alpha_miner()

def test_heuristic():
    #: this function is used for debugging PLEASE IGNORE
    preprocessor = preprocessing.Preprocessing()
    #path =  "data/train_day_joined.csv"
    #path = "data/BPI_Challenge_2019.xes"
    path = "data/Hospital_log.xes"
    #path = "data/dummy.csv"
    #path =  "data/running.csv"
    #preprocessor.handle_import(False,path,"case_id", "timetamp", "activity") 
    preprocessor.handle_import(True,path,"case:concept:name", "time:timestamp", "concept:name") 
    train, test = preprocessor.split_train_test(.9)

    neural_manager = nn_manager.NNManagement()
    # select cuda or not
    neural_manager.config.cuda = True 
    neural_manager.config.absolute_frequency_distribution = preprocessor.absolute_frequency_distribution
    neural_manager.config.number_classes = preprocessor.number_classes
    neural_manager.config.emb_dim =2000 
    neural_manager.config.mlp_dim=2000 
    neural_manager.config.hid_dim= 2000
    neural_manager.config.epochs=1
    neural_manager.load_data(train, test, preprocessor.case_id_key, preprocessor.case_timestamp_key, preprocessor.case_activity_key)
    neural_manager.train()
    neural_manager.config.activity_le = preprocessor.activity_le
    neural_manager.config.case_id_le = preprocessor.case_id_le
    neural_manager.config.exponent = preprocessor.exponent

    pmm = process_model_manager.ProcessModelManager(
        preprocessor.event_df, 
        neural_manager.model, 
        neural_manager.config,
        preprocessor.case_activity_key,
        preprocessor.case_id_key,
        preprocessor.case_timestamp_key
    )
    pmm.unencoded_df = preprocessor.unencoded_df 

    pmm.generate_predictive_log(non_stop=False, upper =100, random_cuts=True, new_log_path = "generated_predicted_df.csv")
    pmm.heuristic_miner(view = False, dependency_threshold=0.8, and_threshold=0.8, loop_two_threshold=0.8,  path = "projects/first Prophet/petri_nets/awesome_heristic.pnml")


    initial = str(pmm.initial_marking)
    final  = str(pmm.final_marking)

    petri_net_config = {
        "initial_marking": initial,
        "final_marking":    final 
    }
    print(petri_net_config)
    print(type(pmm.initial_marking))
    with open("projects/first Prophet/petri_nets/awesome_heristic.pnml.json","w") as f: 
        json.dump(petri_net_config, f)
    print(f"achieved fitness: {pmm.conformance_checking_token_based_replay()}")
    #print(f"achieved fitness: {pmm.conformance_checking_alignments()}")
    #pmm.visualize()

def test_import_model():
    #: this function is used for debugging PLEASE IGNORE
    preprocessor = preprocessing.Preprocessing()
    is_xes  = False
    
    #path =  "data/train_day_joined.csv"
    #path = "data/BPI_Challenge_2019.xes"
    path = "data/Hospital_log.xes"
    #path = "data/dummy.csv"
    #path =  "data/running.csv"
     
    if is_xes:
        #preprocessor.import_event_log_xes(path , "case:concept:name", "concept:name", "time:timestamp")# hospital
        preprocessor.import_event_log_xes(path , "case:concept:name", "concept:name", "time:timestamp")# bpi 2019
    else:
        preprocessor.import_event_log_csv(path , "case_id", "activity", "timestamp", ',')
    train, test = preprocessor.split_train_test(.9)
    neural_manager1 = nn_manager.NNManagement()
    neural_manager1.config.cuda = True 
    neural_manager1.config.absolute_frequency_distribution = preprocessor.absolute_frequency_distribution
    neural_manager1.config.number_classes = preprocessor.number_classes
    neural_manager1.load_data(train, test, preprocessor.case_id_key, preprocessor.case_timestamp_key, preprocessor.case_activity_key)
    neural_manager1.train()
    neural_manager1.config.activity_le = preprocessor.activity_le
    neural_manager1.config.case_id_le = preprocessor.case_id_le

    neural_manager1.export_nn_model()

    neural_manager = nn_manager.NNManagement()
    neural_manager.import_nn_model("model.pt")
    #: unfortunately the le object is lost when exported.
    neural_manager.config.activity_le = preprocessor.activity_le
    neural_manager.config.case_id_le = preprocessor.case_id_le
    neural_manager.config.exponent = preprocessor.exponent
    pmm = process_model_manager.ProcessModelManager(
        preprocessor.event_df, 
        neural_manager.model, 
        neural_manager.config,
        preprocessor.case_activity_key,
        preprocessor.case_id_key,
        preprocessor.case_timestamp_key
    )
    
    pmm.generate_predictive_log_tail_cut()
   

def show_petri_net(): 
    path = "projects/first Prophet/petri_nets/p_net1.pnml"
    net, initial_marking, final_marking = pm4py.read_pnml(path)
    #pm4py.view_petri_net(net, initial_marking, final_marking, format = 'html')
    pm4py.save_vis_petri_net(net, initial_marking, final_marking, 'petri_net.png')

    print("done")


if __name__=="__main__": 
    #test_our()
    #test_import_model()
    #test_random_search(2)
    #test_grid_search()
    #test_single_prediction()
    #test_multiple_prediction()
    #test_process_model_manager_random_cut()
    #test_process_model_manager_random_cut_nontstop()
    #test_process_model_manager_random_cut()
    #test_end_activities()
    #test_process_model_manager_tail_cut()
    #test_heuristic()
    #HL_shorter()
    #show_petri_net()
    app.run(port = SERVER_PORT,debug=True) #: run the flask server
