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
        path = "data/running-example.csv"
        cls.preprocessor.import_event_log_csv(path , "case_id", "activity", "timestamp", ";")

        # setup the nn manager
        train, test = cls.preprocessor.split_train_test(.5)
        cls.nn_manager = NNManagement(None)
        cls.nn_manager.config.absolute_frequency_distribution = cls.preprocessor.absolute_frequency_distribution
        cls.nn_manager.config.number_classes = cls.preprocessor.number_classes
        cls.nn_manager.config.case_id_le = cls.preprocessor.case_id_le
        cls.nn_manager.config.activity_le = cls.preprocessor.activity_le
        cls.nn_manager.config.exponent = cls.preprocessor.exponent
        cls.nn_manager.config.seq_len = 3
        cls.nn_manager.load_data(train, test, cls.preprocessor.case_id_key, cls.preprocessor.case_timestamp_key, cls.preprocessor.case_activity_key)

    def test_train_time_limit(self):
        """
        test if the train time limit is respected
        """
        self.nn_manager.config.train_time_limit = 0 
        with self.assertRaises(exceptions.TrainTimeLimitExceeded):
            self.nn_manager.train()

        self.nn_manager.config.train_time_limit = None

    def test_get_training_statistics(self):
        """
        test if one can get statistics before training
        """
        with self.assertRaises(exceptions.ModelNotTrainedYet):
            nn_manager = NNManagement(None)
            nn_manager.get_training_statistics()
    
    def test_evaluate_loss_values(self):
        """
        check if the loss values lie in the correct range.
        check if f1 can be calculated from acc and recall
        """
        self.nn_manager.train()
        _, acc, recall, f1 = self.nn_manager.evaluate()
        self.assertTrue(0 <= acc <= 1)
        self.assertTrue(0 <= recall <= 1)
        self.assertTrue(0 <= f1 <= 1)

        f1_calculated = 2 * ((acc * recall) / (acc + recall))
        self.assertAlmostEqual(f1, f1_calculated, places=5)
        

        
    


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