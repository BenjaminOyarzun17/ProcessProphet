import sys
import os
import unittest
import pandas as pd

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from server import ProcessModelManager
from server import NNManagement


class TestHelperFunctions(unittest.TestCase):
    """
    The test class for helper functions in the ProcessModelManager class
    - check_too_short
    - decode_df
    - handle_nat
    """

    def setUp(self):
        # Load the model
        nn_manager = NNManagement()
        # load model
        nn_manager.import_nn_model("test_model.pt")
        model = nn_manager.model
        config = nn_manager.model.config

        # Load the event log
        event_df = pd.read_csv('data/running-example.csv', sep=';')
        case_activity_key = 'activity'
        case_id_key = 'case_id'
        case_timestamp_key = 'timestamp'

        # Create instance of ProcessModelManager
        self.pmm = ProcessModelManager(event_df, model, config, case_activity_key, case_id_key, case_timestamp_key)

    def test_check_too_short(self):
        """
        Test the check_too_short function
        """
        old_conf_len = self.pmm.config.seq_len
        self.pmm.config.seq_len = 3

        # To short cases
        case1 = [[1, 2], [4, 5, 6]]
        case2 = [[1, 2, 3], [4, 5], [7, 8, 9]]
        case3 = [[1, 2, 3], [4, 5, 6], []]
        short_cases = [case1, case2, case3]

        # Not too short cases
        case4 = [[1, 2, 3], [4, 5, 6]]
        case5 = [[1, 2, 3, 3, 4, 2], [4, 5, 6, 2], [7, 8, 9]]
        case6 = [[1, 2, 3], [4, 5, 6], [7, 8, 9, 1, 1, 1]]
        not_short_cases = [case4, case5, case6]

        for case in short_cases:
            self.assertTrue(self.pmm.check_too_short(case))
        
        for case in not_short_cases:
            self.assertFalse(self.pmm.check_too_short(case))

        self.pmm.config.seq_len = old_conf_len
        

    def test_decode_df(self):
        # Test with a dataframe
        df = pd.DataFrame({
            'case_id': [1, 2, 3],
            'activity': [1, 2, 3],
            'timestamp': pd.date_range(start='1/1/2022', periods=3)
        })

        # first encode the df
        

        decoded_df = self.pmm.decode_df(df)
        self.assertEqual(decoded_df['activity'].tolist(), ['Activity 1', 'Activity 2', 'Activity 3'])

    def test_handle_nat(self):
        # Test with a dataframe that contains NaT values
        df = pd.DataFrame({
            'case_id': [1, 2, 3],
            'activity': [1, 2, 3],
            'timestamp': [pd.Timestamp('2022-01-01'), pd.NaT, pd.Timestamp('2022-01-03')]
        })
        handled_df = self.pmm.handle_nat(df)
        self.assertFalse(handled_df['timestamp'].isnull().any())

if __name__ == '__main__':
    unittest.main()