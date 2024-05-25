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







def test_data_loader():
    preprocessor = Preprocessing()
    preprocessor.import_event_log_csv("data/test_day.csv"
                                        , "case_id", "activity", "timestamp", ',')
    
    nn_manager = NNManagement()
    train, test, no_classes = preprocessor.split_train_test(.9)
    
    train_set = ATMDataset(nn_manager.config ,train, preprocessor.case_id_key,   preprocessor.case_timestamp_key, preprocessor.case_activity_key) 

    train_loader = DataLoader(train_set, batch_size=nn_manager.config.batch_size, shuffle=True, collate_fn=ATMDataset.to_features)

    for i, batch in enumerate(train_loader):
        a, b = batch
        print(a.shape)
        print("b: ", b.shape)






app = Flask(__name__)
app.register_blueprint(routes)



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
    nn_manager.config.our_implementation = True
    nn_manager.train(train, test, preprocessor.case_id_key, preprocessor.case_timestamp_key, preprocessor.case_activity_key, preprocessor.number_classes)
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
    train, test, no_classes, absolute_frequency_distribution = preprocessor.split_train_test(.9)

    #stats_in_json = nn_manager.get_training_statistics()
    # Define the hyperparameter search space [lower_bound, upper_bound, step_size]
    sp= {
        "hidden_dim": [500, 2000, 500],
        "lstm_dim": [500, 2000,  500],
        "emb_dim": [500, 2000, 500]
    }
    nn_manager = NNManagement()
    acc = 0
    current_params = ()
    for i in range(sp["hidden_dim"][0], sp["hidden_dim"][1], sp["hidden_dim"][2]): 
        nn_manager.config.hid_dim =i 
        for j in range(sp["lstm_dim"][0], sp["lstm_dim"][1], sp["lstm_dim"][2]): 
            nn_manager.config.lstm_dim=j
            for k in range(sp["emb_dim"][0], sp["emb_dim"][1], sp["emb_dim"][2]):
                nn_manager.config.emb_dim=k
                nn_manager.config.absolute_frequency_distribution =  preprocessor.absolute_frequency_distribution
                nn_manager.config.our_implementation = True
                nn_manager.train(train, test, preprocessor.case_id_key, preprocessor.case_timestamp_key, preprocessor.case_activity_key, preprocessor.number_classes)
                if nn_manager.acc> acc: 
                    acc = nn_manager.acc
                    current_params= (i,j,k)
                    logger_grid_search.debug("best accuracy: ")
                    logger_grid_search.debug(acc)
                    logger_grid_search.debug("best current: ")
                    logger_grid_search.debug(current_params)
    

    logger_grid_search.debug(f"best acc {acc} ")
    logger_grid_search.debug(f"best params {current_params}")
    return (acc, current_params)


    
def test_random_search(iterations):
    preprocessor = Preprocessing()
    is_xes  =True
    #path =  "data/train_day_joined.csv"
    path = "data/BPI_Challenge_2019.xes"
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
    acc = 0
    current_params = ()
    nn_manager.config.absolute_frequency_distribution = preprocessor.absolute_frequency_distribution
    for i in range(iterations): 
        a=random.randint(sp["hidden_dim"][0], sp["hidden_dim"][1])
        b=random.randint(sp["lstm_dim"][0], sp["lstm_dim"][1])
        c=  random.randint(sp["emb_dim"][0], sp["emb_dim"][1])
        nn_manager.config.hid_dim = a
        nn_manager.config.emb_dim= b
        nn_manager.config.lstm_dim=c
        nn_manager.train(train, test, preprocessor.case_id_key, preprocessor.case_timestamp_key, preprocessor.case_activity_key, preprocessor.no_classes)
        if nn_manager.acc> acc: 
            acc = nn_manager.acc
            current_params= (a,b,c )
            logging.info("best accuracy: ")
            logging.info(acc)
            logging.info("best current: ")
            logging.info(current_params)

    print(f"best acc {acc}") 
    print(f"best params {current_params}") 
    return (acc, current_params)



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
    nn_manager.config.our_implementation = True
    nn_manager.train(train, test, preprocessor.case_id_key, preprocessor.case_timestamp_key, preprocessor.case_activity_key, preprocessor.number_classes)

    pm = PredictionManager(
        nn_manager.model, 
        preprocessor.case_id_key, 
        preprocessor.case_activity_key, 
        preprocessor.case_timestamp_key, 
        nn_manager.config
    )
    pm.config = nn_manager.config
    pm.model = nn_manager.model
    dummy = pm.get_dummy_process(preprocessor.event_df, preprocessor.case_id_key)
    print(pm.single_prediction_dataframe(dummy))


  


def test_multiple_prediction():
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
    nn_manager.config.our_implementation = True
    nn_manager.config.activity_le = preprocessor.activity_le
    nn_manager.config.case_id_le = preprocessor.case_id_le
    nn_manager.train(train, test, preprocessor.case_id_key, preprocessor.case_timestamp_key, preprocessor.case_activity_key, preprocessor.number_classes)

    dummy = pm.get_dummy_process(preprocessor.event_df, preprocessor.case_id_key)
    pm = PredictionManager( nn_manager.model, preprocessor.case_id_key, preprocessor.case_activity_key, preprocessor.case_timestamp_key, nn_manager.config)
    pm.multiple_prediction_dataframe(
        2, 
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
    nn_manager.config.our_implementation = True
    nn_manager.train(train, test, preprocessor.case_id_key, preprocessor.case_timestamp_key, preprocessor.case_activity_key, preprocessor.number_classes)
    nn_manager.config.activity_le = preprocessor.activity_le
    nn_manager.config.case_id_le = preprocessor.case_id_le
    
    pmm = ProcessModelManager(
        preprocessor.event_df, 
        nn_manager.model, 
        nn_manager.config,
        preprocessor.case_activity_key,
        preprocessor.case_id_key,
        preprocessor.case_timestamp_key
    )
    pmm.end_activities = preprocessor.find_end_activities()
    pmm.generate_predictive_log_random_cut_until_end(100)


def test_process_model_manager_random_cut():
    preprocessor = Preprocessing()
    is_xes  = True
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

    nn_manager = NNManagement()
    # select cuda or not
    nn_manager.config.cuda = True 
    nn_manager.config.absolute_frequency_distribution = preprocessor.absolute_frequency_distribution
    nn_manager.config.our_implementation = True
    nn_manager.train(train, test, preprocessor.case_id_key, preprocessor.case_timestamp_key, preprocessor.case_activity_key, preprocessor.number_classes)
    nn_manager.config.activity_le = preprocessor.activity_le
    nn_manager.config.case_id_le = preprocessor.case_id_le
    
    pmm = ProcessModelManager(
        preprocessor.event_df, 
        nn_manager.model, 
        nn_manager.config,
        preprocessor.case_activity_key,
        preprocessor.case_id_key,
        preprocessor.case_timestamp_key
    )
    pmm.generate_predictive_log_random_cut(100)


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
    nn_manager.config.our_implementation = True
    nn_manager.train(train, test, preprocessor.case_id_key, preprocessor.case_timestamp_key, preprocessor.case_activity_key, preprocessor.number_classes)
    nn_manager.config.activity_le = preprocessor.activity_le
    nn_manager.config.case_id_le = preprocessor.case_id_le
    

    pmm = ProcessModelManager(
        preprocessor.event_df, 
        nn_manager.model, 
        nn_manager.config,
        preprocessor.case_activity_key,
        preprocessor.case_id_key,
        preprocessor.case_timestamp_key
    )
    pmm.generate_predictive_log_tail_cut()


def test_import_model():
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


    nn_manager = NNManagement()
    nn_manager.import_nn_model("model.pt")
    nn_manager.config.cuda = True 
    nn_manager.config.absolute_frequency_distribution = preprocessor.absolute_frequency_distribution
    nn_manager.config.our_implementation = True

    pm = PredictionManager(
        nn_manager.model, 
        preprocessor.case_id_key, 
        preprocessor.case_activity_key, 
        preprocessor.case_timestamp_key, 
        nn_manager.config
    )

    dummy = pm.get_dummy_process(preprocessor.event_df, preprocessor.case_id_key)

    pm.multiple_prediction_dataframe(
        2, 
        2, 
        dummy, 
        preprocessor.case_id_key, 
        preprocessor.case_activity_key, 
        preprocessor.case_timestamp_key, 
        nn_manager.config
    )



if __name__=="__main__": 
    #test_our()
    #test_import_model()
    #test_random_search(2)
    #test_single_prediction()
    #test_process_model_manager_random_cut()
    test_process_model_manager_random_cut_nontstop()
    #test_end_activities()
    #app.run()
