import unittest
from preprocessing import *
from nn_manager import *




class TestImportXESFunction(unittest.TestCase):
    """
    - test order of the rows
    - test datatype of 3 columns
    specification: 
    - input: path, 3 column names
    - outputs/effects: 
        dataframe
        event log object 
        --> could assume it works, so dont test it
    """

    @classmethod
    def setUpClass(cls):
        #: run this to import the data once 
        cls.preprocessor= Preprocessing()
        path = "../data/BPI_Challenge_2019.xes"
        cls.preprocessor.import_event_log_xes(path , "case:concept:name", "concept:name", "time:timestamp")# bpi 2019

    def test_no_nan(self):
        """check if there are no nans"""
        count = self.preprocessor.event_df.isna().sum().sum()
        print(count)
        self.assertEqual(count, 0)

    def test_columns(self):
        """
        test if there are three columns and if they match the input names
        """
        dataframe = self.preprocessor.event_df
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