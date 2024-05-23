from loggers import logger_get_dummy_process,logger_single_prediction , logger_multiple_prediction
from exceptions import ProcessTooShort
from preprocessing import Preprocessing
from ERPP_RMTPP_torch import * 
from torch.utils.data import DataLoader




class PredictionManager: 
    def __init__(self):
        self.model = None
        self.input_df = None


    def get_dummy_process(self, df, case_id_column):
        random_case_id = df[case_id_column].sample(n = 1).values[0]
        dummy = df[df[case_id_column]==random_case_id]
        n_rows = dummy.shape[0]
        if n_rows == 1:
            raise ProcessTooShort()
        logger_get_dummy_process.debug(dummy) 
        #return dummy.iloc[:n_rows-1]
        return dummy.iloc[:10]



    def single_prediction_csv(self, path, case_id, activity_key, timestamp_key, config, sep):
        """
        make one prediction given a csv file path 
        """
        preprocessor = Preprocessing()
        preprocessor.import_event_log_csv(path, case_id, activity_key, timestamp_key, sep)
        encoded_df= preprocessor.event_df 
        self.single_prediction(encoded_df, case_id, activity_key, timestamp_key, config )

    def single_prediction_xes(self, path, case_id, activity_key, timestamp_key, config):
        """
        make one prediction given a xes file path 
        """
        preprocessor = Preprocessing()
        preprocessor.import_event_log_xes(path, case_id, activity_key, timestamp_key )
        encoded_df= preprocessor.event_df 
        self.single_prediction(encoded_df, case_id, activity_key, timestamp_key, config )

    def single_prediction_dataframe(self, df, case_id, activity_key, timestamp_key, config):
        """
        make one prediction given a dataframe 
        """
        preprocessor = Preprocessing()
        preprocessor.import_event_log_dataframe(df, case_id, activity_key, timestamp_key )
        encoded_df= preprocessor.event_df 
        self.single_prediction(encoded_df, case_id, activity_key, timestamp_key, config )


    def single_prediction(self, df, case_id, activity_key, timestamp_key, config ):
        """
        make one prediction 
        """
        step1= ATMDataset(config, df, case_id, timestamp_key, activity_key)
        #: batch size set to one to have one sample per batch.
        step2 = DataLoader(step1, batch_size=1, shuffle=False, collate_fn=ATMDataset.to_features)
        pred_times, pred_events = [], []
        for i, batch in enumerate(step2):   
            pred_time, pred_event = self.model.predict(batch)
            pred_times.append(pred_time)
            pred_events.append(pred_event)
        logger_single_prediction.debug("predicted time:")
        logger_single_prediction.debug(pred_times)
        logger_single_prediction.debug("predicted event:")
        logger_single_prediction.debug(pred_events)
        return pred_times, pred_events


    def multiple_prediction(self, depth, degree,df, case_id, activity_key, timestamp_key, config ):
        """
        make multiple predictions 
        :param depth: how many steps are to be predicted in the future
        :param degree: when moving one step in the future, how
        many possible outputs should be considered
        """
        step1= ATMDataset(config, df, case_id, timestamp_key, activity_key)
        #: batch size set to one to have one sample per batch.
        step2 = DataLoader(step1, batch_size=1, shuffle=False, collate_fn=ATMDataset.to_features)
        pred_times, pred_events = [], []
        for i, batch in enumerate(step2):   
            pred_time, pred_event = self.model.predict_sorted(batch)
            pred_times.append(pred_time)
            pred_events.append(pred_event)
        pred_times= pred_times[-1] #we are only interested in the last one.
        pred_events = pred_events[-1]


        return pred_times, pred_events



    def multiple_prediction_csv(self):
        """
        make multiple predictions given a csv
        """
        pass

    def multiple_prediction_xes(self):
        """
        make multiple predictions given an xes
        """
        pass

    def multiple_prediction_dataframe(self, depth, degree, df, case_id, activity_key, timestamp_key, config):
        """
        make multiple predictions given a dataframe
        """
        preprocessor = Preprocessing()
        preprocessor.import_event_log_dataframe(df, case_id, activity_key, timestamp_key )
        encoded_df= preprocessor.event_df 
        self.multiple_prediction(depth, degree, encoded_df, case_id, activity_key, timestamp_key, config )

