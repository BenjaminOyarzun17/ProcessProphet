from preprocessing import *
from nn_manager import *



if __name__=="__main__": 

    preprocessor = Preprocessing()
    is_xes = True
    if is_xes:
        preprocessor.import_event_log_xes("data/Hospital_log.xes")
        print(preprocessor.event_df.head())
    else:
        preprocessor.import_event_log_csv("data/running-example_1.csv"
                                        , "case_id", "activity", "timestamp", ';')
    train, test, no_classes = preprocessor.split_train_test(.9)
    nn_manager = NNManagement()
    nn_manager.train(train, test, preprocessor.case_id_key, preprocessor.case_timestamp_key, preprocessor.case_activity_key, no_classes)
    stats_in_json = nn_manager.get_training_statistics()
    #nn_manager.model.predict_get_sorted(pass)