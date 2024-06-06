import unittest
import sys
import os
import requests
import pandas as pd


#: this module intends to tests the server routes


SERVER_NAME="localhost"
SERVER_PORT=8080


class TestRemoveDuplicates(unittest.TestCase):
    """
    Test class for training and evaluating the model
    """
    @classmethod
    def setUpClass(cls):
        cls.route = "http://{SERVER_NAME}:{SERVER_PORT}/remove_duplicates"
        cls.dummy_params = {
            "path_to_log": "tests/dummy_data/Hospital_log_mini.xes" , 
            "case_id": "case:concept:name", 
            "activity_key":  "concept:name", 
            "timestamp_key": "timestamp:key" , 
            "is_xes": True, 
            "save_path": "tests/dummy_data/Hospital_log_mini_nodup.csv" ,
            "sep": ","
        }
        #: run this to import the data once 
     

    def test_xes_wrong(self):
        response = requests.post(
            f"http://{SERVER_NAME}:{SERVER_PORT}/remove_duplicates", 
            json= {
                "is_xes": "definitely not a boolean"
            },
            timeout =6000
        )
        self.assertEqual(400, response.status_code)
        data = response.json()
        self.assertDictEqual({"error": "is_xes should be boolean"}, data)
        response = requests.post(
            self.route,
            json= {
                "is_xes": "True"
            },
            timeout =6000
        )
        self.assertEqual(400, response.status_code)
        data = response.json()
        self.assertDictEqual({"error": "is_xes should be boolean"}, data)

    def test_path_does_not_exist(self):
        cpy = self.dummy_params
        cpy["path_to_log"] = "definitely not a path"
        response = requests.post(
            self.route,
            json= cpy, 
            timeout =6000
        )
        self.assertEqual(400, response.status_code)
        cpy = self.dummy_params
        cpy["path_to_log"] = "definitely not a path"
        response = requests.post(
            f"http://{SERVER_NAME}:{SERVER_PORT}/remove_duplicates", 
            json= cpy
        )
        self.assertEqual(400, response.status_code)

    def test_model_already_exists(self):
        cpy = self.dummy_params
        cpy["save_path"] = "tests/dummy_data/Hospital_log_mini.xes"
        response = requests.post(
            self.route,
            json= cpy 
        )
        self.assertEqual(400, response.status_code)
        data = response.json()
        self.assertDictEqual({"error": f"{cpy['save_path']} already exists..."}, data)
    def test_wrong_column_name(self):
        cpy = self.dummy_params
        cpy["case_id"] = "definitely not a case id column"
        response = requests.post(
            self.route,
            json= cpy 
        )
        self.assertEqual(400, response.status_code)
        data = response.json()
        self.assertEqual("error while importing",data["description"] )

    def test_no_duplicates(self): 
        response = requests.post(
            self.route,
            json= self.dummy_params
        )
        self.assertEqual(200, response.status_code)
        data = response.json()
        self.assertEqual(self.dummy_params['save_path'], data['save_path'])
        file_exists = os.path.isfile(self.dummy_params['save_path'])
        self.assertTrue(file_exists)
        res = pd.read_csv(self.dummy_params['save_path'], sep = ',')
        len_res = len(res)
        no_dup = res.drop_duplicates()
        self.assertEqual(len_res, len(no_dup))
        






if __name__ == "__main__":
    unittest.main()