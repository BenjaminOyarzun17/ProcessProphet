from preprocessing import *
from nn_manager import *
from flask import Flask
from server_routes import routes
import time
import logging
#from ray import tune
from functools import partial
import random
from loggers import logger_grid_search, logger_random_search
from prediction_manager import PredictionManager
from process_model_manager import ProcessModelManager







app = Flask(__name__)
app.register_blueprint(routes)


def dummy():
    preprocessor = Preprocessing()
    is_xes = False
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

    pm = PredictionManager(None, "case_id", "activity", "timestamp", None)

    dummy  = pm.get_dummy_process(preprocessor.event_df,"case_id" )
    dummy.to_csv("CLI/input_logs/dummy.csv",sep = ',' )



def test_end_activities():
    preprocessor = Preprocessing()
    is_xes = False
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
    print(preprocessor.find_end_activities())

 
def test_our():
    preprocessor = Preprocessing()
    is_xes = False
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
    
    
    train, test = preprocessor.split_train_test(.7)
    nn_manager = NNManagement()
    # select cuda or not
    nn_manager.config.cuda = True 
    nn_manager.config.absolute_frequency_distribution = preprocessor.absolute_frequency_distribution
    nn_manager.config.number_classes = preprocessor.number_classes
    nn_manager.load_data(train, test, preprocessor.case_id_key, preprocessor.case_timestamp_key, preprocessor.case_activity_key)
    nn_manager.train()
    stats_in_json = nn_manager.get_training_statistics()
    nn_manager.export_nn_model()

def test_grid_search():

    preprocessor = Preprocessing()
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


    #stats_in_json = nn_manager.get_training_statistics()
    # Define the hyperparameter search space [lower_bound, upper_bound, step_size]
    sp= {
        "hidden_dim": [500, 2000, 500],
        "lstm_dim": [500, 2000,  500],
        "emb_dim": [500, 2000, 500]
    }
    nn_manager = NNManagement()
    nn_manager.config.absolute_frequency_distribution = preprocessor.absolute_frequency_distribution
    nn_manager.config.cuda = True 
    nn_manager.config.number_classes = preprocessor.number_classes
    nn_manager.grid_search(train, test, sp, preprocessor.case_id_key, preprocessor.case_timestamp_key, preprocessor.case_activity_key)
    


    
def test_random_search(iterations):
    preprocessor = Preprocessing()
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
    nn_manager = NNManagement()
    nn_manager.config.absolute_frequency_distribution = preprocessor.absolute_frequency_distribution
    nn_manager.config.cuda = True 
    nn_manager.config.number_classes = preprocessor.number_classes
    nn_manager.random_search(train, test, sp, iterations, preprocessor.case_id_key, preprocessor.case_timestamp_key, preprocessor.case_activity_key)




def test_single_prediction():
    preprocessor = Preprocessing()
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

    nn_manager = NNManagement()
    # select cuda or not
    nn_manager.config.cuda = True 
    nn_manager.config.absolute_frequency_distribution = preprocessor.absolute_frequency_distribution
    nn_manager.config.number_classes = preprocessor.number_classes
    nn_manager.load_data(train, test, preprocessor.case_id_key, preprocessor.case_timestamp_key, preprocessor.case_activity_key)
    nn_manager.train()

    pm = PredictionManager(
        nn_manager.model, 
        preprocessor.case_id_key, 
        preprocessor.case_activity_key, 
        preprocessor.case_timestamp_key, 
        nn_manager.config
    )
    dummy = pm.get_dummy_process(preprocessor.event_df, preprocessor.case_id_key)
    print(pm.single_prediction_dataframe(dummy))


  


def test_multiple_prediction():
    preprocessor = Preprocessing()
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

    nn_manager = NNManagement()
    # select cuda or not
    nn_manager.config.cuda = True 
    nn_manager.config.absolute_frequency_distribution = preprocessor.absolute_frequency_distribution
    nn_manager.config.activity_le = preprocessor.activity_le
    nn_manager.config.case_id_le = preprocessor.case_id_le
    nn_manager.config.number_classes = preprocessor.number_classes
    nn_manager.config.exponent = preprocessor.exponent
    nn_manager.load_data(train, test, preprocessor.case_id_key, preprocessor.case_timestamp_key, preprocessor.case_activity_key)
    nn_manager.train()

    pm = PredictionManager( nn_manager.model, preprocessor.case_id_key, preprocessor.case_activity_key, preprocessor.case_timestamp_key, nn_manager.config)
    dummy = pm.get_dummy_process(preprocessor.event_df, preprocessor.case_id_key)
    pm.multiple_prediction_dataframe(
        3, 
        2, 
        dummy  
    )

def test_process_model_manager_random_cut_nontstop():
    preprocessor = Preprocessing()
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

    nn_manager = NNManagement()
    # select cuda or not
    nn_manager.config.cuda = True 
    nn_manager.config.absolute_frequency_distribution = preprocessor.absolute_frequency_distribution
    nn_manager.config.number_classes = preprocessor.number_classes
    nn_manager.load_data(train, test, preprocessor.case_id_key, preprocessor.case_timestamp_key, preprocessor.case_activity_key)
    nn_manager.train()
    nn_manager.config.activity_le = preprocessor.activity_le
    nn_manager.config.case_id_le = preprocessor.case_id_le
    nn_manager.config.exponent = preprocessor.exponent

    pmm = ProcessModelManager(
        preprocessor.event_df, 
        nn_manager.model, 
        nn_manager.config,
        preprocessor.case_activity_key,
        preprocessor.case_id_key,
        preprocessor.case_timestamp_key
    )
    pmm.end_activities = preprocessor.find_end_activities()
    pmm.generate_predictive_log(non_stop=True, upper =30, random_cuts=True, new_log_path="generated_predicted_df.csv")
    pmm.decode_df()
    pmm.predictive_df.to_csv("generated_predicted_df.csv")


def test_process_model_manager_random_cut():
    preprocessor = Preprocessing()
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

    nn_manager = NNManagement()
    # select cuda or not
    nn_manager.config.cuda = True 
    nn_manager.config.absolute_frequency_distribution = preprocessor.absolute_frequency_distribution
    nn_manager.config.number_classes = preprocessor.number_classes
    nn_manager.load_data(train, test, preprocessor.case_id_key, preprocessor.case_timestamp_key, preprocessor.case_activity_key)
    nn_manager.train()
    nn_manager.config.activity_le = preprocessor.activity_le
    nn_manager.config.case_id_le = preprocessor.case_id_le
    nn_manager.config.exponent = preprocessor.exponent
    
    pmm = ProcessModelManager(
        preprocessor.event_df, 
        nn_manager.model, 
        nn_manager.config,
        preprocessor.case_activity_key,
        preprocessor.case_id_key,
        preprocessor.case_timestamp_key
    )
    pmm.generate_predictive_log(non_stop=False, upper =100, random_cuts=True, new_log_path = "generated_predicted_df.csv")
    pmm.decode_df()
    pmm.predictive_df.to_csv("generated_predicted_df.csv")

def test_process_model_manager_tail_cut():
    preprocessor = Preprocessing()
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

    nn_manager = NNManagement()
    # select cuda or not
    nn_manager.config.cuda = True 
    nn_manager.config.absolute_frequency_distribution = preprocessor.absolute_frequency_distribution
    nn_manager.config.number_classes = preprocessor.number_classes
    nn_manager.load_data(train, test, preprocessor.case_id_key, preprocessor.case_timestamp_key, preprocessor.case_activity_key)
    nn_manager.train()
    nn_manager.config.activity_le = preprocessor.activity_le
    nn_manager.config.case_id_le = preprocessor.case_id_le
    nn_manager.config.exponent = preprocessor.exponent
    

    pmm = ProcessModelManager(
        preprocessor.event_df, 
        nn_manager.model, 
        nn_manager.config,
        preprocessor.case_activity_key,
        preprocessor.case_id_key,
        preprocessor.case_timestamp_key
    )

    pmm.generate_predictive_log(non_stop=False, random_cuts=False, cut_length = 3)


def test_alpha_miner():

    preprocessor = Preprocessing()
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

    nn_manager = NNManagement()
    # select cuda or not
    nn_manager.config.cuda = True 
    nn_manager.config.absolute_frequency_distribution = preprocessor.absolute_frequency_distribution
    """
    nn_manager.config.epochs = 15
    nn_manager.config.emb_dim =1000
    nn_manager.config.hid_dim =1000
    nn_manager.config.mlp_dim =1000
    """
    nn_manager.config.number_classes = preprocessor.number_classes
    nn_manager.load_data(train, test, preprocessor.case_id_key, preprocessor.case_timestamp_key, preprocessor.case_activity_key)
    nn_manager.train()
    nn_manager.config.activity_le = preprocessor.activity_le
    nn_manager.config.case_id_le = preprocessor.case_id_le
    nn_manager.config.exponent = preprocessor.exponent
    
    pmm = ProcessModelManager(
        preprocessor.event_df, 
        nn_manager.model, 
        nn_manager.config,
        preprocessor.case_activity_key,
        preprocessor.case_id_key,
        preprocessor.case_timestamp_key
    )
    pmm.end_activities = preprocessor.find_end_activities()
    #pmm.generate_predictive_log_random_cut_until_end(100)
    pmm.generate_predictive_log_random_cut(100)
    #pmm.alpha_miner()
    pmm.heuristic_miner(view = True)

def test_heuristic():
    preprocessor = Preprocessing()
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

    nn_manager = NNManagement()
    # select cuda or not
    nn_manager.config.cuda = True 
    nn_manager.config.absolute_frequency_distribution = preprocessor.absolute_frequency_distribution
    nn_manager.config.number_classes = preprocessor.number_classes
    nn_manager.load_data(train, test, preprocessor.case_id_key, preprocessor.case_timestamp_key, preprocessor.case_activity_key)
    nn_manager.train()
    nn_manager.config.activity_le = preprocessor.activity_le
    nn_manager.config.case_id_le = preprocessor.case_id_le
    nn_manager.config.exponent = preprocessor.exponent

    pmm = ProcessModelManager(
        preprocessor.event_df, 
        nn_manager.model, 
        nn_manager.config,
        preprocessor.case_activity_key,
        preprocessor.case_id_key,
        preprocessor.case_timestamp_key
    )
    

    pmm.generate_predictive_log(non_stop=False, upper =100, random_cuts=True, new_log_path = "generated_predicted_df.csv")
    pmm.heuristic_miner(view = True, path = "awesome_heristic.pnml")


def test_import_model():
    preprocessor = Preprocessing()
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
    nn_manager1 = NNManagement()
    nn_manager1.config.cuda = True 
    nn_manager1.config.absolute_frequency_distribution = preprocessor.absolute_frequency_distribution
    nn_manager1.config.number_classes = preprocessor.number_classes
    nn_manager1.load_data(train, test, preprocessor.case_id_key, preprocessor.case_timestamp_key, preprocessor.case_activity_key)
    nn_manager1.train()
    nn_manager1.config.activity_le = preprocessor.activity_le
    nn_manager1.config.case_id_le = preprocessor.case_id_le

    nn_manager1.export_nn_model()

    nn_manager = NNManagement()
    nn_manager.import_nn_model("model.pt")
    #: unfortunately the le object is lost when exported.
    nn_manager.config.activity_le = preprocessor.activity_le
    nn_manager.config.case_id_le = preprocessor.case_id_le
    nn_manager.config.exponent = preprocessor.exponent
    pmm = ProcessModelManager(
        preprocessor.event_df, 
        nn_manager.model, 
        nn_manager.config,
        preprocessor.case_activity_key,
        preprocessor.case_id_key,
        preprocessor.case_timestamp_key
    )
    
    pmm.generate_predictive_log_tail_cut()
   




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
    test_heuristic()
    #dummy()
    #app.run()
