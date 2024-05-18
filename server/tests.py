import unittest
from preprocessing import *
from nn_manager import *




class TestImportXESFunction(unittest.TestCase):
    """
    - test order of the rows
    - test for 3 columns  
    - test datatype of 3 columns
    - check if 3 col names match with log
    - Nan entries = 0
    specification: 
    - input: path, 3 column names
    - outputs/effects: 
        dataframe
        event log object 
        --> could assume it works, so dont test it
    """
    def test_columns(self):
        preprocessor= Preprocessing()
        path = "../data/BPI_Challenge_2019.xes"
        preprocessor.import_event_log_xes(path , "case:concept:name", "concept:name", "time:timestamp")# bpi 2019
        dataframe = preprocessor.event_df
        columns = set(dataframe.columns)
        gold = set(["case:concept:name", "concept:name", "time:timestamp"])
        self.assertSetEqual(gold, columns)



class TestImportCSVFunction(unittest.TestCase):
    def test_positive_number(self):
        self.assertEqual(abs(10), 10)

    def test_negative_number(self):
        self.assertEqual(abs(-10), 10)

    def test_zero(self):
        self.assertEqual(abs(0), 0)



if __name__ == "__main__":
    unittest.main()