from preprocessing import *
from nn_manager import *
from flask import Flask
from server_routes import routes
import time
import logging
#from ray import tune
from functools import partial





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
    
def test_our()    :
    preprocessor = Preprocessing()
    is_xes = True
    #path =  "../data/train_day_joined.csv"
    path = "../data/BPI_Challenge_2019.xes"
    #path = "../data/Hospital_log.xes"
    #path = "../data/dummy.csv"
    #path =  "../data/running.csv"
    
    if is_xes:
        #preprocessor.import_event_log_xes(path , "case:concept:name", "concept:name", "time:timestamp")# hospital
        preprocessor.import_event_log_xes(path , "case:concept:name", "concept:name", "time:timestamp")# bpi 2019
        print(preprocessor.event_df.head())
    else:
        preprocessor.import_event_log_csv(path , "case_id", "activity", "timestamp", ',')
    
    
    train, test, no_classes, absolute_frequency_distribution = preprocessor.split_train_test(.9)
    nn_manager = NNManagement()
    nn_manager.absolute_frequency_distribution = absolute_frequency_distribution
    nn_manager.config.our_implementation = True
    nn_manager.train(train, test, preprocessor.case_id_key, preprocessor.case_timestamp_key, preprocessor.case_activity_key, no_classes)
    stats_in_json = nn_manager.get_training_statistics()
    nn_manager.export_nn_model()

def test_embed():
    preprocessor = Preprocessing()
    is_xes = True
    #path =  "../data/train_day_joined.csv"
    path = "../data/BPI_Challenge_2019.xes"
    #path = "../data/Hospital_log.xes"
    #path = "../data/dummy.csv"
    #path =  "../data/running.csv"
     
    if is_xes:
        #preprocessor.import_event_log_xes(path , "case:concept:name", "concept:name", "time:timestamp")# hospital
        preprocessor.import_event_log_xes(path , "case:concept:name", "concept:name", "time:timestamp")# bpi 2019
        print(preprocessor.event_df.head())
    else:
        preprocessor.import_event_log_csv(path , "case_id", "activity", "timestamp", ',')
    train, test, no_classes, absolute_frequency_distribution = preprocessor.split_train_test(.9)

    for emb_dim in range (50, 1100, 50):
        nn_manager = NNManagement()
        nn_manager.absolute_frequency_distribution = absolute_frequency_distribution
        nn_manager.config.our_implementation = True
        nn_manager.config.emb_dim = emb_dim
        nn_manager.train(train, test, preprocessor.case_id_key, preprocessor.case_timestamp_key, preprocessor.case_activity_key, no_classes)
        stats_in_json = nn_manager.get_training_statistics()
        logging.basicConfig(filename='embed.log', filemode='a')
        logging.info(f"dimension used: {emb_dim}")
        logging.info(stats_in_json)

def test_embed():
    preprocessor = Preprocessing()
    is_xes =True
    #path =  "../data/train_day_joined.csv"
    path = "../data/BPI_Challenge_2019.xes"
    #path = "../data/Hospital_log.xes"
    #path = "../data/dummy.csv"
    #path =  "../data/running.csv"
     
    if is_xes:
        #preprocessor.import_event_log_xes(path , "case:concept:name", "concept:name", "time:timestamp")# hospital
        preprocessor.import_event_log_xes(path , "case:concept:name", "concept:name", "time:timestamp")# bpi 2019
        print(preprocessor.event_df.head())
    else:
        preprocessor.import_event_log_csv(path , "case_id", "activity", "timestamp", ',')
    train, test, no_classes, absolute_frequency_distribution = preprocessor.split_train_test(.9)

    logging.basicConfig(filename='embed.log', level=logging.INFO,  format='%(message)s'  ) 
    logging.info(f"no. classes: {no_classes}")
    for dim in range (1200, 2000, 200):
        nn_manager = NNManagement()
        nn_manager.absolute_frequency_distribution = absolute_frequency_distribution
        nn_manager.config.our_implementation = True
        nn_manager.config.emb_dim = dim
        nn_manager.config.hid_dim= dim
        nn_manager.config.mlp_dim = dim//2
        nn_manager.train(train, test, preprocessor.case_id_key, preprocessor.case_timestamp_key, preprocessor.case_activity_key, no_classes)
        stats_in_json = nn_manager.get_training_statistics()
        logging.info(f"dimension used: {dim}")
        logging.info(stats_in_json)

"""
def test_grid_search():

    preprocessor = Preprocessing()
    is_xes =True
    #path =  "../data/train_day_joined.csv"
    path = "../data/BPI_Challenge_2019.xes"
    #path = "../data/Hospital_log.xes"
    #path = "../data/dummy.csv"
    #path =  "../data/running.csv"
     
    if is_xes:
        #preprocessor.import_event_log_xes(path , "case:concept:name", "concept:name", "time:timestamp")# hospital
        preprocessor.import_event_log_xes(path , "case:concept:name", "concept:name", "time:timestamp")# bpi 2019
        print(preprocessor.event_df.head())
    else:
        preprocessor.import_event_log_csv(path , "case_id", "activity", "timestamp", ',')
    train, test, no_classes, absolute_frequency_distribution = preprocessor.split_train_test(.9)

    nn_manager = NNManagement()
    nn_manager.absolute_frequency_distribution = absolute_frequency_distribution
    nn_manager.config.our_implementation = True
    nn_manager.train(train, test, preprocessor.case_id_key, preprocessor.case_timestamp_key, preprocessor.case_activity_key, no_classes)
    #stats_in_json = nn_manager.get_training_statistics()
    # Define the hyperparameter search space
    search_space = {
        "hidden_dim": tune.grid_search([500, 750, 1000, 1500 , 2000]),
        "lstm_dim": tune.grid_search([500, 750, 1000, 1500 , 2000]),
        "emb_dim": tune.grid_search([500, 750, 1000, 1500 , 2000])
    }

    # Run the grid search
    analysis = tune.run(
        partial(nn_manager.train, train, test, preprocessor.case_id_key, preprocessor.case_timestamp_key, preprocessor.case_activity_key, no_classes),
        config=search_space,
        resources_per_trial={"cpu": 0, "gpu": 1},  
        num_samples=1
    )
    print("best params: ", analysis.best_config)

"""



def test_their()    :
    preprocessor = Preprocessing()
    is_xes = False
    path =  "../data/train_day_joined.csv"
    if is_xes:
        preprocessor.import_event_log_xes(path)
        print(preprocessor.event_df.head())
    else:
        preprocessor.import_event_log_csv(path 
                                        , "case_id", "activity", "timestamp", ',')
    
    
    train, test, no_classes = preprocessor.split_train_test(.9)
    nn_manager = NNManagement()
    nn_manager.config.our_implementation = False
    nn_manager.train(train, test, preprocessor.case_id_key, preprocessor.case_timestamp_key, preprocessor.case_activity_key, no_classes)
    stats_in_json = nn_manager.get_training_statistics()

if __name__=="__main__": 
    #test_embed() 
    test_our()
    #test_their()
    #nn_manager.model.predict_get_sorted(pass)
    
    #app.run()