import unittest
import os
import sys
# needed for the import of the server module:
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from server.preprocessing import Preprocessing
from server.nn_manager import NNManagement
import server.exceptions as exceptions


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
        

class TestImportExportModel(unittest.TestCase):
    """
    Test class for importing and exporting NN models
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
        cls.nn_manager.train()

    def test_export_nn_model(self):
        """
        test if the model can be exported
        """
        self.nn_manager.export_nn_model("test_model.pt")
        self.assertTrue(os.path.exists("test_model.pt"))

    def test_import_nn_model(self):
        """
        test if the model can be imported
        """
        nn_manager = NNManagement(None)
        nn_manager.import_nn_model("test_model.pt")
        self.assertTrue(nn_manager.model is not None)

    def test_export_import_model(self):
        """
        test export and import of the model 
        """
        self.nn_manager.export_nn_model("test_model")
        nn_manager = NNManagement(None)
        nn_manager.import_nn_model("test_model")
        self.assertTrue(nn_manager.model is not None)


class TestHyperparameterTuning(unittest.TestCase):
    """
    Test class for Grid Search and Random Search
    """
    @classmethod
    def setUpClass(cls):
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

    def test_grid_search(self):
        """
        test if grid search can be performed
        """
        search_params = {
            "hidden_dim": [30, 50, 10],
            "mlp_dim": [10, 30, 10],
            "emb_dim": [30, 50, 10],
        }
        self.nn_manager.grid_search(search_params)

    def test_random_search(self):
        """
        test if random search can be performed
        """
        search_params = {
            "hidden_dim": [1, 50],
            "mlp_dim": [10, 30],
            "emb_dim": [30, 50],
        }
        self.nn_manager.random_search(search_params, 3)


if __name__ == "__main__":
    unittest.main()