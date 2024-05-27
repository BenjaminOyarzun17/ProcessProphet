import unittest
from preprocessing import *
from nn_manager import *
import numpy as np
import exceptions


class TestTrainEvaluate(unittest.TestCase):
    """
    tests: 
    - check if there are no nan's while training /
    as output of the predictions
    - check if train/test are empty, raise an exception.
    - raise an exception if cols dont exist. 

    specification: 
    - input: 
        - traindata, testdata, all 3 col names,no_classes

    - output/sideeffects:
        - no classes is set
        - nn is trained and tested
        - nn model is saved in the class 
        - acc, recall, f1 are saved in class
    """
    @classmethod
    def setUpClass(cls):
        #: run this to import the data once 
        cls.preprocessor= Preprocessing()
        path = "data/running-example.csv" #its smaller, use preferrably.
        cls.preprocessor.import_event_log_csv(path , "case_id", "activity", "timestamp", ";")

    def test_train_time_limit(self):
        """
        test if the train time limit is respected
        """
        train, test = self.preprocessor.split_train_test(.5)
        nn_manager = NNManagement(None)
        nn_manager.config.absolute_frequency_distribution = self.preprocessor.absolute_frequency_distribution
        nn_manager.config.number_classes = self.preprocessor.number_classes
        nn_manager.config.train_time_limit = 0.1
        nn_manager.load_data(train, test, self.preprocessor.case_id_key, self.preprocessor.case_timestamp_key, self.preprocessor.case_activity_key)

        with self.assertRaises(exceptions.TrainTimeLimitExceeded):
            nn_manager.train()



class TestImportXESFunction(unittest.TestCase):
    """
    Test class for the `Preprocessing.import_event_log_xes` method
    """

    @classmethod
    def setUpClass(cls):
        #: run this to import the data once 
        cls.preprocessor= Preprocessing()
        path = "data/Hospital_log.xes" #its smaller, use preferrably.
        cls.preprocessor.import_event_log_xes(path , "case:concept:name", "concept:name", "time:timestamp")

    def test_no_nan(self):
        """check if there are no nans"""
        count = self.preprocessor.event_df.isna().sum().sum()
        print(count)
        self.assertEqual(count, 0)
        
    def test_column_types(self):
        column_types = list(self.preprocessor.event_df.dtypes) 
        column_types = list(map(str, column_types))
        self.assertListEqual(["float64"]*3 , column_types)

   
    
    def test_columns(self):
        """
        test if there are three columns and if they match the input names
        """
        dataframe = self.preprocessor.event_df
        columns = set(dataframe.columns)
        gold = set(["case:concept:name", "concept:name", "time:timestamp"])
        self.assertSetEqual(gold, columns)

    def test_correct_subfolder(self):
        pass #: TODO (waiting for CLI)


if __name__ == "__main__":
    unittest.main()