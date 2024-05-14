from preprocessing import *
from nn_manager import *

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



if __name__=="__main__": 
    preprocessor = Preprocessing()
    is_xes = False
    if is_xes:
        preprocessor.import_event_log_xes("data/Hospital_log.xes")
        print(preprocessor.event_df.head())
    else:
        preprocessor.import_event_log_csv("data/test_day.csv"
                                        , "case_id", "activity", "timestamp", ',')
    train, test, no_classes = preprocessor.split_train_test(.9)
    nn_manager = NNManagement()
    nn_manager.train(train, test, preprocessor.case_id_key, preprocessor.case_timestamp_key, preprocessor.case_activity_key, no_classes)
    stats_in_json = nn_manager.get_training_statistics()

    #nn_manager.model.predict_get_sorted(pass)
