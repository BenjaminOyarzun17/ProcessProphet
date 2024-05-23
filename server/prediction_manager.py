from loggers import logger_get_dummy_process,logger_single_prediction 
from exceptions import ProcessTooShort
from preprocessing import Preprocessing




class PredictionManager: 
    def __init__(self):
        self.model = None


    def get_dummy_process(self, df, case_id_column):
        random_case_id = df[case_id_column].sample(n = 1).values[0]
        dummy = df[df[case_id_column]==random_case_id]
        n_rows = dummy.rows[0]
        if n_rows == 1:
            raise ProcessTooShort()
        logger_get_dummy_process.debug(dummy) 
        return dummy.iloc[:n_rows-1]



    def single_prediction_csv(self):
        """
        make one prediction given a csv file path 
        """
        pass

    def single_prediction_xes(self):
        """
        make one prediction given a xes file path 
        """
        pass

    def single_prediction_dataframe(self, df, case_id, activity_key, timestamp_key):
        """
        make one prediction given a dataframe 
        """
        preprocessor = Preprocessing()
        preprocessor.import_event_log_dataframe(df, case_id, activity_key, timestamp_key )
        preprocessor.encode_df_columns()
        time, event = self.model.predict()
        logger_single_prediction.debug("predicted time:")
        logger_single_prediction.debug(time)
        logger_single_prediction.debug("predicted event:")
        logger_single_prediction.debug(event)
        return time, event 
        


    def single_prediction(self):
        """
        make one prediction 
        """
        pass



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

    def multiple_prediction_dataframe(self):
        """
        make multiple predictions given a dataframe
        """
        pass

    def multiple_prediction(self):
        """
        make multiple predictions 
        """
        pass