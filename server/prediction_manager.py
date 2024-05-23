from loggers import logger_get_dummy_process





class PredictionManager: 
    def __init__(self):
        pass


    def get_dummy_process(self, df, case_id_column):
        random_case_id = df[case_id_column].sample(n = 1).values[0]
        dummy = df[df[case_id_column]==random_case_id]
        logger_get_dummy_process.debug(dummy) 



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

    def single_prediction_dataframe(self):
        """
        make one prediction given a dataframe 
        """
        pass

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