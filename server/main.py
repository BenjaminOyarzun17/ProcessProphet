from preprocessing import *
from nn_manager import *



if __name__=="__main__": 
    preprocessor = Preprocessing()
    preprocessor.import_event_log_csv("../data/running-example.csv"
                                      , "case_id", "activity", "timestamp", ';')
    train, test, no_classes = preprocessor.split_train_test(.8)

    nn_manager = NNManagement()
    nn_manager.train(train, test, preprocessor.case_id_key, preprocessor.case_timestamp_key, preprocessor.case_activity_key, no_classes)