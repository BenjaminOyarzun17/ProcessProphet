import unittest
import sys
import os
import requests
import pandas as pd


#: this module intends to tests the server routes


SERVER_NAME="localhost"
SERVER_PORT=8080


class TestPreprocessingRoutes(unittest.TestCase):
    """
    tests the three preprocessing routes
    """
    @classmethod
    def setUpClass(cls):
        cls.routes = [
            "http://{SERVER_NAME}:{SERVER_PORT}/remove_duplicates",
            "http://{SERVER_NAME}:{SERVER_PORT}/add_unique_start_end",
            "http://{SERVER_NAME}:{SERVER_PORT}/replace_with_mode"
                      ]
        cls.dummy_params = {
            "path_to_log": "tests/dummy_data/Hospital_log_mini.xes" , 
            "case_id": "case:concept:name", 
            "activity_key":  "concept:name", 
            "timestamp_key": "timestamp:key" , 
            "is_xes": True, 
            "save_path": "tests/dummy_data/Hospital_log_mini_" ,
            "sep": ","
        }
        #: run this to import the data once 
     

    def test_xes_wrong(self):
        for route in self.routes:
            response = requests.post(
                route,
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
        for route in self.routes:
            cpy = self.dummy_params.copy()
            cpy["path_to_log"] = "definitely not a path"
            response = requests.post(
                route,
                json= cpy, 
                timeout =6000
            )
            self.assertEqual(400, response.status_code)
            cpy = self.dummy_params.copy()
            cpy["path_to_log"] = "definitely not a path"
            response = requests.post(
                f"http://{SERVER_NAME}:{SERVER_PORT}/remove_duplicates", 
                json= cpy
            )
            self.assertEqual(400, response.status_code)

    def test_model_already_exists(self):
        for route in self.routes:
            cpy = self.dummy_params.copy()
            cpy["save_path"] = "tests/dummy_data/Hospital_log_mini.xes"
            response = requests.post(
                route,
                json= cpy 
            )
            self.assertEqual(400, response.status_code)
            data = response.json()
            self.assertDictEqual({"error": f"{cpy['save_path']} already exists..."}, data)
    def test_wrong_column_name(self):
        for route in self.routes:
            cpy = self.dummy_params
            cpy["case_id"] = "definitely not a case id column"
            response = requests.post(
                route, 
                json= cpy 
            )
            self.assertEqual(400, response.status_code)
            data = response.json()
            self.assertEqual("error while importing",data["description"] )

    def test_no_duplicates(self): 
        cpy = self.dummy_params.copy
        new_path  = self.dummy_params['save_path']+self.routes[0]+".csv"

        cpy['save_path']= new_path
        response = requests.post(
            self.routes[0],
            json= cpy
        )
        self.assertEqual(200, response.status_code)
        data = response.json()
        self.assertEqual(new_path, data['save_path'])
        file_exists = os.path.isfile(self.dummy_params['save_path'])
        self.assertTrue(file_exists)
        res = pd.read_csv(new_path, sep = ',')
        len_res = len(res)
        no_dup = res.drop_duplicates()
        self.assertEqual(len_res, len(no_dup))
        os.remove(self.dummy_params["save_path"])
        

    def test_unique_start_end(self): 
        response = requests.post(
            self.routes[1],
            json= self.dummy_params
        )


    def test_replace_with_mode(self): 
        cpy = self.dummy_params.copy
        new_path  = self.dummy_params['save_path']+self.routes[2]+".csv"

        cpy['save_path']= new_path
        response = requests.post(
            self.routes[2],
            json= cpy
        )
        self.assertEqual(200, response.status_code)
        data = response.json()
        self.assertEqual(self.dummy_params['save_path'], data['save_path'])
        file_exists = os.path.isfile(self.dummy_params['save_path'])
        self.assertTrue(file_exists)
        os.remove(self.dummy_params["save_path"])


    @classmethod
    def tearDownClass(cls) -> None:
        """
        ensure to delete the generated csv's,as they will not be used any more. 
        """
        try:
            os.remove(cls.dummy_params["save_path"]+cls.routes[2]+".csv")
        except: 
            pass
            
        try:
            os.remove(cls.dummy_params["save_path"]+cls.routes[1]+".csv")
        except: 
            pass

        try:
            os.remove(cls.dummy_params["save_path"]+cls.routes[0]+".csv")
        except: 
            pass


if __name__ == "__main__":
    unittest.main()