import unittest
from preprocessing import *
from nn_manager import *
import numpy as np



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
        self.assertListEqual(["string", "string", "datetime64[ns, UTC]"], column_types)

   
    def test_row_order(self):
        """
        test whether the timestamps are in the correct order and 
        TODO: whether the case id's are properly grouped
        """
        case_id_key = self.preprocessor.case_id_key
        case_timestamp_key = self.preprocessor.case_timestamp_key
        curr_case_id = self.preprocessor.event_df.iloc[0][case_id_key]
        curr_timestamp= self.preprocessor.event_df.iloc[0][case_timestamp_key]
        

        no_rows, no_cols = self.preprocessor.event_df.shape
        case_id_memo = {case_id: False for case_id in self.preprocessor.event_df[case_id_key].unique()}
        correct_order=True 
        for i in range(1, no_rows): #iterate over all rows
            next_case_id = self.preprocessor.event_df.iloc[i][case_id_key]
            next_time_stamp = self.preprocessor.event_df.iloc[i][case_timestamp_key]
            if curr_case_id == next_case_id:
                #:if the case id match, test if both conseq. timestamps are in the correct order. 
                if curr_timestamp >  next_time_stamp:
                    correct_order = False
                    break
            else: 
                #:if the case id does not match, check whether it has already been seen.
                if case_id_memo[next_case_id]:
                    correct_order  =False
                    break
                else: 
                    case_id_memo[next_case_id] = True
            curr_case_id = next_case_id
            curr_timestamp = next_time_stamp

        self.assertTrue(correct_order)



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

class TestImportCSVFunction(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        #: run this to import the data once 
        cls.preprocessor= Preprocessing()
        path = "data/running-example.csv" #its smaller, use preferrably.
        cls.preprocessor.import_event_log_csv(path , "case_id", "activity", "timestamp", ";")

    def test_no_nan(self):
        """check if there are no nans"""
        count = self.preprocessor.event_df.isna().sum().sum()
        print(count)
        self.assertEqual(count, 0)
        
    def test_column_types(self):
        column_types = list(self.preprocessor.event_df.dtypes) 
        column_types = list(map(str, column_types))
        self.assertListEqual(["string", "string", "datetime64[ns, UTC]"], column_types)

   
    def test_row_order(self):
        """
        test whether the timestamps are in the correct order and 
        TODO: whether the case id's are properly grouped
        """
        case_id_key = self.preprocessor.case_id_key
        case_timestamp_key = self.preprocessor.case_timestamp_key
        curr_case_id = self.preprocessor.event_df.iloc[0][case_id_key]
        curr_timestamp= self.preprocessor.event_df.iloc[0][case_timestamp_key]
        

        no_rows, no_cols = self.preprocessor.event_df.shape
        case_id_memo = {case_id: False for case_id in self.preprocessor.event_df[case_id_key].unique()}
        correct_order=True 
        for i in range(1, no_rows): #iterate over all rows
            next_case_id = self.preprocessor.event_df.iloc[i][case_id_key]
            next_time_stamp = self.preprocessor.event_df.iloc[i][case_timestamp_key]
            if curr_case_id == next_case_id:
                #:if the case id match, test if both conseq. timestamps are in the correct order. 
                if curr_timestamp >  next_time_stamp:
                    correct_order = False
                    break
            else: 
                #:if the case id does not match, check whether it has already been seen.
                if case_id_memo[next_case_id]:
                    correct_order  =False
                    break
                else: 
                    case_id_memo[next_case_id] = True
            curr_case_id = next_case_id
            curr_timestamp = next_time_stamp

        self.assertTrue(correct_order)



    def test_columns(self):
        """
        test if there are three columns and if they match the input names
        """
        dataframe = self.preprocessor.event_df
        columns = set(dataframe.columns)
        gold = set(["activity", "case_id", "timestamp"])
        self.assertSetEqual(gold, columns)

    def test_correct_subfolder(self):
        pass #: TODO (waiting for CLI)

    """
    missing tests: 
    - check if Counter does contain all the classes
    - check if all the classes have been counted
    - check correspondance between the encoded caseids and the origonal versions
    -   ditto for encoded markers 
    """

class TestSplitTrainTest(unittest.TestCase):
    """
    correctness criteria: 
    -  check if raise TrainPercentageTooHigh is generated for a very small  
    event log; also test it for train_percentage = 1
    - TODO: check the NaN solution correctness (the right exponent selection) or
    any other implemented solution
    - check that all columns are of type float

    specification: 
    - input: train percentage

    - out/sideeffects 
    """
    pass


class TestTrainEvaluate(unittest.TestCase):
    """
    tests: 
    - check if there are no nan's while training /
    as output of the predictions
    - check if train/test are empty, raise an exception.
    - raise an exception if cols dont exist. 
    - 

    specification: 
    - input: 
        - traindata, testdata, all 3 col names,no_classes

    - output/sideeffects:
        - no classes is set
        - nn is trained and tested
        - nn model is saved in the class 
        - acc, recall, f1 are saved in class

    """



if __name__ == "__main__":
    unittest.main()