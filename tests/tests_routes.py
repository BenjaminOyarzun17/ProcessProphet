import unittest
import sys
import os
import requests
import pandas as pd


#: this module tests the server routes

class TestTrainingRoutes(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        SERVER_NAME="localhost"
        SERVER_PORT=8080

        cls.routes = [
            f"http://{SERVER_NAME}:{SERVER_PORT}/multiple_prediction",
            f"http://{SERVER_NAME}:{SERVER_PORT}/single_prediction",
                      ]
        cls.dummy_params = {
            "path_to_log": "tests/dummy_data/partial_input.csv" , 
            "case_id": "case:concept:name", 
            "activity_key":  "concept:name", 
            "timestamp_key": "time:timestamp" , 
            "sep": ",", 
            "path_to_model": "tests/dummy_data/prediction_test_model.pt", 
            "config":"tests/dummy_data/prediction_test_model.config.pt", 
            "depth": 3, 
            "degree": 1, 
            "prediction_file_name":"tests/dummy_data/prediction_test_prediction.json" 
        }

    def test_paths(self): 
        for route in self.routes:
            for file in ["path_to_log", "config", "path_to_model"]:
                cpy = self.dummy_params.copy()
                cpy[file]= "definitely not a path in any OS"
            
                response = requests.post(
                    route,
                    json= cpy, 
                    timeout =6000
                )
                self.assertEqual(400, response.status_code)
                data = response.json()
                self.assertEqual(data["error"], "one required path does not exist")


        cpy = self.dummy_params.copy()
        cpy["prediction_file_name"]= "tests/dummy_data/DONOTDELETE.pt" 
        response = requests.post(
            self.routes[0],
            json= cpy, 
            timeout =6000
        )
        self.assertEqual(400, response.status_code)
        data = response.json()
        self.assertEqual(data["error"], "the target path for the new file already exists")
         
              
    def test_wrong_integer_params(self): 
        cpy = self.dummy_params.copy()
        
      
        for param in ["degree", "path"]: 
            cpy[param]= "definitely not an int value"
            response = requests.post(
                self.routes[0],
                json= cpy, 
                timeout =6000
            )
            self.assertEqual(400, response.status_code)
            data = response.json()
            self.assertEqual(data["error"], "an integer param was set to another type")
    def test_file_creation(self):
        for index, route in enumerate(self.routes): 
            response = requests.post(
                self.routes[0],
                json= self.params, 
                timeout =6000
            )
            self.assertEqual(response.status_code, 200)
            if index == 0:  
                exists = os.path.isfile(self.params["prediction_file_name"])
                self.asserTrue(exists)
            

    @classmethod
    def tearDownClass(cls) -> None:
        try:
            os.remove("tests/dummy_data/prediction_test_prediction.json")
        except: 
            pass
    


class TestTrainingRoutes(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        SERVER_NAME="localhost"
        SERVER_PORT=8080

        cls.routes = [
            f"http://{SERVER_NAME}:{SERVER_PORT}/train_nn",
            f"http://{SERVER_NAME}:{SERVER_PORT}/random_search",
            f"http://{SERVER_NAME}:{SERVER_PORT}/grid_search" 
                      ]
        cls.dummy_params = {
            "path_to_log": "tests/dummy_data/Hospital_log_mini1.xes" , 
            "case_id": "case:concept:name", 
            "activity_key":  "concept:name", 
            "timestamp_key": "time:timestamp" , 
            "is_xes": True, 
            "sep": ",", 
            "cuda": True, 
            "model_path": "tests/dummy_data/dummy_output_model.pt", 
            "split": 0.5, 
            "seq_len":10, 
            "emb_dim":1000, 
            "hid_dim":1000, 
            "mlp_dim":1000, 
            "lr":0.001, 
            "batch_size":200, 
            "epochs": 2, 
            "search_params": {
                "hid_dim": [100, 200],
                "emb_dim":[100, 200],
                "mlp_dim":[100, 200]
            },
            "iterations": 1
        }
        try:
            os.remove("tests/dummy_data/dummy_output_model.pt")
        except: 
            pass
        try:
            os.remove("tests/dummy_data/dummy_output_model.config.json")
        except: 
            pass

        try:
            os.remove("tests/dummy_data/some_other_new_model.pt")
        except: 
            pass
        try:
            os.remove("tests/dummy_data/some_other_new_model.config.json")
        except: 
            pass

    def test_wrong_booleans(self):
        for route in self.routes:
            cpy = self.dummy_params.copy()
            if "grid" in route: 
                cpy["search_params"]["emb_dim"] = [100, 200, 100]
                cpy["search_params"]["mlp_dim"]  = [100, 200, 100]
                cpy["search_params"]["hid_dim"] = [100, 200, 100]
            for param in ["cuda", "is_xes"]:
                cpy[param]= "definitely not a boolean"
                response = requests.post(
                    route,
                    json= cpy, 
                    timeout =6000
                )
                self.assertEqual(400, response.status_code)
                data = response.json()
                self.assertEqual(data["error"], "a boolean param was set to another type")
            
    def test_wrong_integer_params(self): 
        for route in self.routes:
            cpy = self.dummy_params.copy()
            if "grid" in route: 
                cpy["search_params"]["emb_dim"]=[100,200,100]
                cpy["search_params"]["mlp_dim"]=[100,200,100]
                cpy["search_params"]["hid_dim"]=[100,200,100]
            if "train" in route:
                integer_params = ["seq_len", "emb_dim", "hid_dim", "mlp_dim", "batch_size", "epochs"]
            elif "random" in route:
                integer_params = ["seq_len",  "batch_size", "epochs", "iterations"]
            elif "grid" in route:
                integer_params = ["seq_len",  "batch_size", "epochs"]
            for param in integer_params: 
                cpy[param]= "definitely not an int value"
                response = requests.post(
                    route,
                    json= cpy, 
                    timeout =6000
                )
                self.assertEqual(400, response.status_code)
                data = response.json()
                self.assertEqual(data["error"], "an integer param was set to another type")
    

    def test_search_params(self): 
        for route in self.routes[1:]:
            cpy = self.dummy_params.copy()
            if "grid" in route: 
                cpy["search_params"]["emb_dim"]=[100,200,100]
                cpy["search_params"]["mlp_dim"]=[100,200,100]
                cpy["search_params"]["hid_dim"]=[100,200,100]
            cpy["search_params"]= {
                "emb_dim": [],
                "hid_dim":[]
            }
            response = requests.post(
                route,
                json= cpy, 
                timeout =6000
            )
            self.assertEqual(400, response.status_code)
            data = response.json()
            self.assertEqual(data["error"], "missing key in search params")


            cpy = self.dummy_params.copy()
            cpy["search_params"]= {
                "emb_dim": [],
                "hid_dim":[], 
                "mlp_dim":[]
            }
            response = requests.post(
                route,
                json= cpy, 
                timeout =6000
            )
            self.assertEqual(400, response.status_code)
            data = response.json()
            self.assertEqual(data["error"], "search param(s) missing")


            cpy = self.dummy_params.copy()
            if "grid" in route:
                cpy["search_params"]= {
                    "emb_dim": [100, 200, "watermelon"],
                    "hid_dim":[100, 200, 100], 
                    "mlp_dim":[100,200,100]
                }

            elif "random" in route: 
                cpy["search_params"]= {
                    "emb_dim": [100, "rhabababa"],
                    "hid_dim":[100, 200], 
                    "mlp_dim":[100,200]
                }
            response = requests.post(
                route,
                json= cpy, 
                timeout =6000
            )
            self.assertEqual(400, response.status_code)
            data = response.json()
            self.assertEqual(data["error"], "non integer found in search params")






    def test_wrong_float_params(self): 
        for route in self.routes:
            cpy = self.dummy_params.copy()
            if "grid" in route: 
                cpy["search_params"]["emb_dim"]=[100,200,100]
                cpy["search_params"]["mlp_dim"]=[100,200,100]
                cpy["search_params"]["hid_dim"]=[100,200,100]
            float_params = ["lr", "split"]
            for param in float_params: 
                cpy[param]= "definitely not a float value"
                response = requests.post(
                    route,
                    json= cpy, 
                    timeout =6000
                )
                self.assertEqual(400, response.status_code)
                data = response.json()
                self.assertEqual(data["error"], "a float param was set to another type")


    def test_paths(self): 
        for route in self.routes:
            cpy = self.dummy_params.copy()
            cpy["path_to_log"]= "definitely not a path in any OS"
            if "grid" in route: 
                cpy["search_params"]["emb_dim"]=[100,200,100]
                cpy["search_params"]["mlp_dim"]=[100,200,100]
                cpy["search_params"]["hid_dim"]=[100,200,100]
            response = requests.post(
                route,
                json= cpy, 
                timeout =6000
            )
            self.assertEqual(400, response.status_code)
            data = response.json()
            self.assertEqual(data["error"], "one required path does not exist")


        for route in self.routes:
            cpy = self.dummy_params.copy()
            cpy["model_path"]= "tests/dummy_data/DONOTDELETE.pt" 
            if "grid" in route: 
                cpy["search_params"]["emb_dim"]=[100,200,100]
                cpy["search_params"]["mlp_dim"]=[100,200,100]
                cpy["search_params"]["hid_dim"]=[100,200,100]
            response = requests.post(
                route,
                json= cpy, 
                timeout =6000
            )
            self.assertEqual(400, response.status_code)
            data = response.json()
            self.assertEqual(data["error"], "the target path for the new file already exists")
         
    def test_split_train_error(self):
        for route in self.routes: 
            cpy = self.dummy_params.copy()
            cpy["split"]= 777.777
            if "grid" in route: 
                cpy["search_params"]["emb_dim"]=[100,200,100]
                cpy["search_params"]["mlp_dim"]=[100,200,100]
                cpy["search_params"]["hid_dim"]=[100,200,100]
            elif "random" in route: 
                cpy["search_params"]["emb_dim"]=[100,200]
                cpy["search_params"]["mlp_dim"]=[100,200]
                cpy["search_params"]["hid_dim"]=[100,200]

            response = requests.post(
                route,
                json= cpy, 
                timeout =6000
            )
            self.assertEqual(400, response.status_code)
            data = response.json()
            self.assertEqual(data["error"], "train percentage must be in range (0,1) and should not yield empty sublogs")


    def test_wrong_column(self):
        for route in self.routes: 
            cpy = self.dummy_params.copy()
            cpy["case_id"]= "clearly not a column"
            if "grid" in route: 
                cpy["search_params"]["emb_dim"]=[100,200,100]
                cpy["search_params"]["mlp_dim"]=[100,200,100]
                cpy["search_params"]["hid_dim"]=[100,200,100]
            elif "random" in route: 
                cpy["search_params"]["emb_dim"]=[100,200]
                cpy["search_params"]["mlp_dim"]=[100,200]
                cpy["search_params"]["hid_dim"]=[100,200]
            response = requests.post(
                route,
                json= cpy, 
                timeout =6000
            )
            self.assertEqual(400, response.status_code)
            data = response.json()
            self.assertEqual(data["description"], "error while importing")


    def test_training_results(self): 
        for route in self.routes: 
            cpy = self.dummy_params.copy()
            cpy["model_path"] = "tests/dummy_data/some_other_new_model.pt"
            if "grid" in route: 
                cpy["search_params"]["emb_dim"]=[100,200,100]
                cpy["search_params"]["mlp_dim"]=[100,200,100]
                cpy["search_params"]["hid_dim"]=[100,200,100]
            elif "random" in route: 
                cpy["search_params"]["emb_dim"]=[100,200]
                cpy["search_params"]["mlp_dim"]=[100,200]
                cpy["search_params"]["hid_dim"]=[100,200]
            response = requests.post(
                route,
                json= cpy, 
                timeout =6000
            )
            data = response.json()
            print(data)
            self.assertEqual(200, response.status_code, cpy)

            model_exists= os.path.isfile(cpy["model_path"])
            config_exists= os.path.isfile(cpy["model_path"][:-3]+".config.json")
            self.assertTrue(model_exists and config_exists)
            self.assertTrue("training_statistics" in data.keys() or "acc" in data.keys())
            os.remove("tests/dummy_data/some_other_new_model.pt")
            os.remove("tests/dummy_data/some_other_new_model.config.json")


    @classmethod
    def tearDownClass(cls) -> None:
        try:
            os.remove("tests/dummy_data/some_other_new_model.pt")
        except: 
            pass
        try:
            os.remove("tests/dummy_data/some_other_new_model.config.json")
        except: 
            pass

class TestPreprocessingRoutes(unittest.TestCase):
    #:tests the three preprocessing routes
    @classmethod
    def setUpClass(cls):
        SERVER_NAME="localhost"
        SERVER_PORT=8080

        cls.routes = [
            f"http://{SERVER_NAME}:{SERVER_PORT}/remove_duplicates",
            f"http://{SERVER_NAME}:{SERVER_PORT}/add_unique_start_end",
            f"http://{SERVER_NAME}:{SERVER_PORT}/replace_with_mode"
                      ]
        cls.dummy_params = {
            "path_to_log": "tests/dummy_data/Hospital_log_mini.xes" , 
            "case_id": "case:concept:name", 
            "activity_key":  "concept:name", 
            "timestamp_key": "time:timestamp" , 
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
            self.assertDictEqual({"error": "a boolean param was set to another type"}, data)
            response = requests.post(
                route,
                json= {
                    "is_xes": "True" #this is not aboolean xd
                },
                timeout =6000
            )
            self.assertEqual(400, response.status_code)
            data = response.json()
            self.assertDictEqual({"error": "a boolean param was set to another type"}, data)

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
                self.routes[0],
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
            self.assertDictEqual({"error": f"the target path for the new file already exists"}, data)
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
        cpy = self.dummy_params.copy()
        new_path  = self.dummy_params['save_path']+"dup"+".csv"

        cpy['save_path']= new_path
        response = requests.post(
            self.routes[0],
            json= cpy
        )
        self.assertEqual(200, response.status_code)
        data = response.json()
        self.assertEqual(new_path, data['save_path'])
        file_exists = os.path.isfile(new_path)
        self.assertTrue(file_exists)
        res = pd.read_csv(new_path, sep = ',')
        len_res = len(res)
        no_dup = res.drop_duplicates()
        self.assertEqual(len_res, len(no_dup))
        

    def test_unique_start_end(self): 
        cpy = self.dummy_params.copy()
        new_path  = self.dummy_params['save_path']+"unique"+".csv"
        cpy['save_path']= new_path
        response = requests.post(
            self.routes[1],
            json= cpy
        )
        data = response.json()
        self.assertEqual(200, response.status_code)
        self.assertEqual(new_path, data['save_path'])
        file_exists = os.path.isfile(new_path)
        self.assertTrue(file_exists)
        df = pd.read_csv(new_path, sep = ',')
        case_ids = df["case:concept:name"].unique()
        check = True
        start= df[df["case:concept:name"]==case_ids[0]].iloc[0]["concept:name"]
        end = df[df["case:concept:name"]==case_ids[0]].iloc[-1]["concept:name"]
        for case_id in case_ids: 
            sub_df = df[df["case:concept:name"]==case_id]
            sub_df = sub_df["concept:name"]
            if sub_df.iloc[0] != start or sub_df.iloc[-1] != end: 
                check = False
                break
        self.assertTrue(check)


    def test_replace_with_mode(self): 
        cpy = self.dummy_params.copy()
        new_path  = self.dummy_params['save_path']+"mode"+".csv"

        cpy['save_path']= new_path
        response = requests.post(
            self.routes[2],
            json= cpy
        )
        self.assertEqual(200, response.status_code)
        data = response.json()
        self.assertEqual(new_path, data['save_path'])
        file_exists = os.path.isfile(new_path)
        self.assertTrue(file_exists)
        df = pd.read_csv(new_path, sep = ',')
        self.assertEqual(0, df.isna().sum().sum() )


    @classmethod
    def tearDownClass(cls) -> None:
        #:ensure to delete the generated csv's,as they will not be used any more. 
        try:
            os.remove(cls.dummy_params["save_path"]+"mode"+".csv")
        except: 
            pass

        try:
            os.remove(cls.dummy_params["save_path"]+"unique"+".csv")
        except: 
            pass

        try:
            os.remove(cls.dummy_params["save_path"]+"dup"+".csv")
        except: 
            pass




if __name__ == "__main__":
    unittest.main()