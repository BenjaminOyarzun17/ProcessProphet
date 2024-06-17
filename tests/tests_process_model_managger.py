import sys
import os
import unittest
import pandas as pd
import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from server import ProcessModelManager
from server import NNManagement


class TestHelperFunctions(unittest.TestCase):
    """
    The test class for helper functions in the ProcessModelManager class
    - check_too_short
    - decode_df
    - handle_nat
    - compute_fitness
    """
    @classmethod
    def setUpClass(cls):
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
        cls.pmm = ProcessModelManager(event_df, model, config, case_activity_key, case_id_key, case_timestamp_key)

    # def test_check_too_short(self):
    #     """
    #     Test the check_too_short function
    #     """
    #     old_conf_len = self.pmm.config.seq_len
    #     self.pmm.config.seq_len = 3

    #     # To short cases
    #     case1 = [[1, 2], [4, 5, 6]]
    #     case2 = [[1, 2, 3], [4, 5], [7, 8, 9]]
    #     case3 = [[1, 2, 3], [4, 5, 6], []]
    #     short_cases = [case1, case2, case3]

    #     # Not too short cases
    #     case4 = [[1, 2, 3], [4, 5, 6]]
    #     case5 = [[1, 2, 3, 3, 4, 2], [4, 5, 6, 2], [7, 8, 9]]
    #     case6 = [[1, 2, 3], [4, 5, 6], [7, 8, 9, 1, 1, 1]]
    #     not_short_cases = [case4, case5, case6]

    #     for case in short_cases:
    #         self.assertTrue(self.pmm.check_too_short(case))
        
    #     for case in not_short_cases:
    #         self.assertFalse(self.pmm.check_too_short(case))

    #     self.pmm.config.seq_len = old_conf_len
        

    # def test_decode_df(self):
    #     """
    #     Test the decode_df function
    #     """
    #     # setup Label Encoders
    #     class ActivityEncoder:
    #         def inverse_transform(self, activities):
    #             return ['Activity ' + str(activity) for activity in activities]
            
    #     class CaseIdEncoder:
    #         def inverse_transform(self, case_ids):
    #             return ['Case ' + str(case_id) for case_id in case_ids]
            
    #     old_activity_le = self.pmm.config.activity_le
    #     old_case_id_le = self.pmm.config.case_id_le
    #     self.pmm.config.activity_le = ActivityEncoder()
    #     self.pmm.config.case_id_le = CaseIdEncoder()

    #     old_exponent = self.pmm.config.exponent
    #     self.pmm.config.exponent = 0

    #     # define test cases
    #     case1 = pd.DataFrame({
    #         'case_id': [1, 2, 3],
    #         'activity': [1, 2, 3],
    #         'timestamp': [1, 2, 3]
    #         # 'timestamp': pd.date_range(start='1/1/2022', periods=3)
    #     })
    #     expected_case1 = pd.DataFrame({
    #         'case_id': ['Case 1', 'Case 2', 'Case 3'],
    #         'activity': ['Activity 1', 'Activity 2', 'Activity 3'],
    #         'timestamp': pd.Series([1, 2, 3]).astype('datetime64[ns, UTC]')
    #     })

    #     case2 = pd.DataFrame({
    #         'case_id': [4, 5, 6],
    #         'activity': [4, 5, 6],
    #         'timestamp': [1, 2, 3]
    #     })
    #     expected_case2 = pd.DataFrame({
    #         'case_id': ['Case 4', 'Case 5', 'Case 6'],
    #         'activity': ['Activity 4', 'Activity 5', 'Activity 6'],
    #         'timestamp': pd.Series([1, 2, 3]).astype('datetime64[ns, UTC]')
    #     })

    #     case3 = pd.DataFrame({
    #         'case_id': [1],
    #         'activity': [1],
    #         'timestamp': [1]
    #     })
    #     expected_case3 = pd.DataFrame({
    #         'case_id': ['Case 1'],
    #         'activity': ['Activity 1'],
    #         'timestamp': pd.Series([1]).astype('datetime64[ns, UTC]')
    #     })

    #     case4 = pd.DataFrame({
    #         'case_id': [1, 2],
    #         'activity': [1, 2],
    #         'timestamp': [1, 2]
    #     })
    #     expected_case4 = pd.DataFrame({
    #         'case_id': ['Case 1', 'Case 2'],
    #         'activity': ['Activity 1', 'Activity 2'],
    #         'timestamp': pd.Series([1, 2]).astype('datetime64[ns, UTC]')
    #     })

    #     case5 = pd.DataFrame({
    #         'case_id': [1, 1, 1, 8, 3, 3, 3, 3, 4, 3, 1],
    #         'activity': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11],
    #         'timestamp': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11]
    #     })
    #     expected_case5 = pd.DataFrame({
    #         'case_id': ['Case 1', 'Case 1', 'Case 1', 'Case 8', 'Case 3', 'Case 3', 'Case 3', 'Case 3', 'Case 4', 'Case 3', 'Case 1'],
    #         'activity': ['Activity 1', 'Activity 2', 'Activity 3', 'Activity 4', 'Activity 5', 'Activity 6', 'Activity 7', 'Activity 8', 'Activity 9', 'Activity 10', 'Activity 11'],
    #         'timestamp': pd.Series([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11]).astype('datetime64[ns, UTC]')
    #     })
        
    #     cases = [case1, case2, case3, case4, case5]
    #     expected_cases = [expected_case1, expected_case2, expected_case3, expected_case4, expected_case5]

    #     for i, case in enumerate(cases):
    #         decoded_df = self.pmm.decode_df(case)
    #         self.assertTrue(decoded_df.equals(expected_cases[i]))

    #     # Reset the config
    #     self.pmm.config.activity_le = old_activity_le
    #     self.pmm.config.case_id_le = old_case_id_le
    #     self.pmm.config.exponent = old_exponent


    def test_handle_nat(self):
        """
        Test the handle_nat function
        """
        # Test with a dataframe that contains NaT values
        case1 = pd.DataFrame({
            'case_id': [1, 1, 1],
            'activity': [1, 2, 3],
            'timestamp': [pd.Timestamp('2022-01-01'), pd.NaT, pd.Timestamp('2022-01-03')]
        })
        case2 = pd.DataFrame({
            'case_id': [1, 1, 1],
            'activity': [4, 5, 6],
            'timestamp': [pd.Timestamp('2022-01-01'), pd.Timestamp('2022-01-02'), pd.NaT]
        })
        case3 = pd.DataFrame({
            'case_id': [1, 1],
            'activity': [1, 3],
            'timestamp': [pd.NaT, pd.Timestamp('2022-01-01')]
        })
        for case in [case1, case2, case3]:
            handled_df = self.pmm.handle_nat(case)
            self.assertFalse(handled_df['timestamp'].isnull().any())

    def test_compute_fitness(self):
        """
        Test the compute_fitness function
        """
        # Define test cases
        case1 = {"missing_tokens": 0, "consumed_tokens": 3, "remaining_tokens": 0, "produced_tokens": 3}
        case2 = {"missing_tokens": 1, "consumed_tokens": 5, "remaining_tokens": 1, "produced_tokens": 4}
        case3 = {"missing_tokens": 2, "consumed_tokens": 3, "remaining_tokens": 2, "produced_tokens": 3}    
        cases = [case1, case2, case3]
        # Expected results
        res = [1, 0.775, 0.3333333]
        
        for i, case in enumerate(cases):
            fitness = self.pmm.compute_fitness([case])
            self.assertAlmostEqual(fitness, res[i], places=3)

class TestTailCutter(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
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
        cls.pmm = ProcessModelManager(event_df, model, config, case_activity_key, case_id_key, case_timestamp_key)

    def test_tail_cutter(self):
        test_cases = [
            {
                "case_id_counts": pd.Series(np.arange(1, 11), index=np.arange(10)),  # Start from 1
                "min_tail_length": 3,
                "expected_case_id_counts_length": 7,
                "expected_cuts_length": 3,
                "expected_input_sequences_length": 3,
            },
            {
                "case_id_counts": pd.Series(np.arange(1, 6), index=np.arange(5)),  # Start from 1
                "min_tail_length": 2,
                "expected_case_id_counts_length": 3,
                "expected_cuts_length": 2,
                "expected_input_sequences_length": 2,
            },
            {
                "case_id_counts": pd.Series(np.array([1]), index=np.arange(1)),  # Only one element with value 1
                "min_tail_length": 1,
                "expected_case_id_counts_length": 0,
                "expected_cuts_length": 1,
                "expected_input_sequences_length": 1,
            },
        ]

        for test_case in test_cases:
            cuts = {}
            input_sequences = []

            case_id_counts, cuts, input_sequences = self.pmm.tail_cutter(
                test_case["case_id_counts"], 
                test_case["min_tail_length"], 
                cuts, 
                input_sequences
            )

            self.assertEqual(len(case_id_counts), test_case["expected_case_id_counts_length"])
            self.assertEqual(len(cuts), test_case["expected_cuts_length"])
            self.assertEqual(len(input_sequences), test_case["expected_input_sequences_length"])

if __name__ == '__main__':
    unittest.main()