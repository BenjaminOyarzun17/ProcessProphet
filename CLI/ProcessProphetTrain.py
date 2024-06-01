import ProcessProphet
import pytermgui as ptg
import json
import requests
import base64
from loggers import logger_set_params_cli
import ProcessProphetStart
from dotenv import load_dotenv
import os



load_dotenv()
SERVER_NAME= os.getenv('SERVER_NAME')
SERVER_PORT= os.getenv('SERVER_PORT')

class ProcessProphetTrain: 
    def __init__(self, pp):
        self.pp = pp #: reference to the PP object 
        #: after creating the object, set the main menu as start screen
        self.pp.switch_window(self.trainer_main_menu())


    def loading(self, message = ""): 
        """
        a loading screen 
        """
        container = ptg.Container(
            "Loading...", 
            message
        )
        window = ptg.Window(container, box="DOUBLE")
        window.center()
        self.pp.switch_window(window)
    

    def return_to_menu(self):
        """
        returns to p.p. start
        """
        pp_start = ProcessProphetStart.ProcessProphetStart(self.pp)



    def start_training(self) : 
        """
        carries out a training request. 

        side effects/ outputs: 
        :return model: a model pt file is saved in the models folder 
        :return config: a models config information for the server is saved in the models folder
        as a json file. 
        the training statistics (accuracy, recall, f1 score)  are 
        displayed on screen. 
        """
        self.loading("preprocessing data...")
        input_logs_path= self.pp.state.input_logs_path

        is_xes = True if self.log_name.value[-3:] == "xes"  else False
        
        params = {
            "path_to_log": f"{input_logs_path}/{self.log_name.value}" , 
            "split": self.split.value, 
            "model_path": f"{self.pp.state.models_path}/{self.model_name.value}", 
            "case_id": self.case_id_key.value, 
            "activity_key":  self.case_activity_key.value, 
            "timestamp_key":  self.case_timestamp_key.value, 
            "cuda": self.cuda.value, 
            "seq_len": self.seq_len.value, 
            "emb_dim": self.emb_dim.value, 
            "hid_dim":self.hid_dim.value, 
            "mlp_dim":self.mlp_dim.value, 
            "lr": self.lr.value, 
            "batch_size": self.batch_size.value, 
            "epochs": self.epochs.value, 
            "is_xes": is_xes
        } 

        response = requests.post(
            f"http://{SERVER_NAME}:{SERVER_PORT}/train_nn", 
            json= params,
            timeout =6000
        )
        if response.status_code == 200: 
            logger_set_params_cli.debug(response.content)
            data = response.json()

            statistics = data["training_statistics"]
            

            container = ptg.Container(
                "training successful", 
                f"accuracy: {statistics['acc']}", 
                f"recall: {statistics['recall']}", 
                f"f1-score: {statistics['f1']}", 
                ptg.Button("training menu", lambda *_: self.pp.switch_window(self.trainer_main_menu())), 
                ptg.Button("action menu", lambda *_:  self.return_to_menu())
            )
        else: 
            container = ptg.Container(
                "training FAILED:",
                ptg.Button("back", lambda *_: self.pp.switch_window(self.set_training_params()))
            )
        window = ptg.Window(container, box="DOUBLE")
        window.center()
        return window


    def set_training_params(self): 
        self.cuda=  ptg.InputField("True", prompt="use cuda: ")
        self.model_name=  ptg.InputField("f.pt", prompt="model name: ")
        self.seq_len=  ptg.InputField("10", prompt="sequence length: ")
        self.emb_dim=  ptg.InputField("32", prompt="embedding dimension: ")
        self.hid_dim=  ptg.InputField("32", prompt="hidden dimension: ")
        self.mlp_dim=  ptg.InputField("16", prompt="mlp dimension: ")
        self.lr=  ptg.InputField("1e-3", prompt="learning rate: ")
        self.batch_size=  ptg.InputField("1024", prompt="batch size: ")
        self.epochs=  ptg.InputField("10", prompt="number of epochs: ")
        self.split=  ptg.InputField("0.9", prompt="split fraction: ")
        self.log_name=  ptg.InputField("BPI_Challenge_2019.xes", prompt="log name: ")
        self.case_id_key=  ptg.InputField("case:concept:name", prompt="case id key: ")
        self.case_activity_key=  ptg.InputField("concept:name", prompt="activity key: ")
        self.case_timestamp_key=  ptg.InputField("time:timestamp", prompt="timestamp key: ")

        container = [
            ptg.Label(f"set parameters for training"),
            self.cuda , 
            self.model_name ,
            self.seq_len ,
            self.emb_dim ,
            self.hid_dim ,
            self.mlp_dim ,
            self.lr ,
            self.batch_size ,
            self.epochs ,
            self.split, 
            self.log_name,
            self.case_id_key, 
            self.case_activity_key, 
            self.case_timestamp_key,
            ptg.Button("continue", lambda *_: self.pp.switch_window(self.start_training()))
        ]

        window = ptg.Window(*container)
        window.center()
        return window

    def start_grid_search(self) : 
        self.loading("preprocessing data...")
        
        input_logs_path= self.pp.state.input_logs_path

        is_xes = True if self.log_name.value[-3:] == "xes"  else False

        sp = {
            "hidden_dim":[self.hidden_dim_lower.value,self.hidden_dim_upper.value, self.hidden_dim_step.value] ,
            "mlp_dim":[self.mlp_dim_lower.value, self.mlp_dim_upper.value, self.mlp_dim_step.value] ,
            "emb_dim":[self.emb_dim_lower.value, self.emb_dim_upper.value, self.emb_dim_step.value] 
        } 
        params = {
            "path_to_log": f"{input_logs_path}/{self.log_name.value}" , 
            "split": self.split.value, 
            "model_path": f"{self.pp.state.models_path}/{self.model_name.value}", 
            "case_id": self.case_id_key.value, 
            "activity_key":  self.case_activity_key.value, 
            "timestamp_key":  self.case_timestamp_key.value, 
            "cuda": self.cuda.value, 
            "seq_len": self.seq_len.value, 
            "lr": self.lr.value, 
            "batch_size": self.batch_size.value, 
            "epochs": self.epochs.value, 
            "is_xes": is_xes,
            "search_params": sp
        } 

        response = requests.post(
            f"http://{SERVER_NAME}:{SERVER_PORT}/grid_search", 
            json= params,
            timeout =6000
        )
        if response.status_code == 200: 
            logger_set_params_cli.debug(response.content)
            data = response.json()

            accuracy = data["acc"]
            


            container = ptg.Container(
                "training successful", 
                f"accuracy: {accuracy}", 
                ptg.Button("training menu", lambda *_: self.pp.switch_window(self.trainer_main_menu())), 
                ptg.Button("action menu", lambda *_:  self.return_to_menu())
            )
        else: 
            container = ptg.Container(
                "training FAILED:", 
                ptg.Button("back", lambda *_: self.pp.switch_window(self.set_grid_search_params()))
            )
        window = ptg.Window(container, box="DOUBLE")
        window.center()
        return window


    def start_random_search(self) : 
        self.loading("preprocessing data...")
        
        input_logs_path= self.pp.state.input_logs_path

        is_xes = True if self.log_name.value[-3:] == "xes"  else False

        sp = {
            "hidden_dim":[self.hidden_dim_lower.value,self.hidden_dim_upper.value] ,
            "mlp_dim":[self.mlp_dim_lower.value, self.mlp_dim_upper.value] ,
            "emb_dim":[self.emb_dim_lower.value, self.emb_dim_upper.value] 
        } 
        params = {
            "path_to_log": f"{input_logs_path}/{self.log_name.value}" , 
            "split": self.split.value, 
            "case_id": self.case_id_key.value, 
            "activity_key":  self.case_activity_key.value, 
            "timestamp_key":  self.case_timestamp_key.value, 
            "cuda": self.cuda.value, 
            "model_path": f"{self.pp.state.models_path}/{self.model_name.value}", 
            "seq_len": self.seq_len.value, 
            "lr": self.lr.value, 
            "batch_size": self.batch_size.value, 
            "epochs": self.epochs.value, 
            "is_xes": is_xes,
            "search_params": sp, 
            "iterations": self.iterations.value
        } 

        response = requests.post(
            f"http://{SERVER_NAME}:{SERVER_PORT}/random_search", 
            json= params,
            timeout =6000
        )
        if response.status_code == 200: 
            logger_set_params_cli.debug(response.content)
            data = response.json()

            accuracy = data["acc"]

            container = ptg.Container(
                "training successful", 
                f"accuracy: {accuracy}", 
                ptg.Button("training menu", lambda *_: self.pp.switch_window(self.trainer_main_menu())), 
                ptg.Button("action menu", lambda *_:  self.return_to_menu())
            )
        else: 
            container = ptg.Container(
                "training FAILED:", 
                ptg.Button("back", lambda *_: self.pp.switch_window(self.set_random_search_params()))
            )
        window = ptg.Window(container, box="DOUBLE")
        window.center()
        return window

    def set_random_search_params(self):
        self.cuda=  ptg.InputField("True", prompt="use cuda: ")
        self.model_name=  ptg.InputField("f.pt", prompt="model name: ")
        self.seq_len=  ptg.InputField("10", prompt="sequence length: ")
        self.lr=  ptg.InputField("1e-3", prompt="learning rate: ")
        self.batch_size=  ptg.InputField("1024", prompt="batch size: ")
        self.epochs= ptg.InputField("10", prompt="number of epochs: ")
        self.split= ptg.InputField("0.9", prompt="split fraction: ")
        self.log_name= ptg.InputField("Hospital_log.xes", prompt="log name: ")
        self.iterations= ptg.InputField("2", prompt="iterations: ")


        self.case_id_key= ptg.InputField("case:concept:name", prompt="case id key: ")
        self.case_activity_key= ptg.InputField("concept:name", prompt="activity key: ")
        self.case_timestamp_key= ptg.InputField("time:timestamp", prompt="timestamp key: ")

        self.hidden_dim_lower= ptg.InputField("100", prompt="hidden dim. lower bound: ")
        self.hidden_dim_upper= ptg.InputField("200", prompt="hidden dim. upper bound: ")
        
        self.mlp_dim_lower= ptg.InputField("100", prompt="mlp dim. lower bound: ")
        self.mlp_dim_upper= ptg.InputField("200", prompt="mlp dim. upper bound: ")


        self.emb_dim_lower= ptg.InputField("100", prompt="emb dim. lower bound: ")
        self.emb_dim_upper= ptg.InputField("200", prompt="emb dim. upper bound: ")

        container = [
            ptg.Label(f"set parameters for grid search"),
            self.cuda , 
            self.model_name ,
            self.seq_len ,
            self.lr ,
            self.batch_size ,
            self.epochs ,
            self.split, 
            self.log_name,
            self.case_id_key, 
            self.case_activity_key, 
            self.case_timestamp_key,
            self.hidden_dim_lower, 
            self.hidden_dim_upper, 
            self.mlp_dim_lower, 
            self.mlp_dim_upper, 
            self.emb_dim_lower, 
            self.emb_dim_upper, 
            self.iterations,
            ptg.Button("continue", lambda *_: self.pp.switch_window(self.start_random_search()))
        ]

        window = ptg.Window(*container)
        window.center()
        return window




    def set_grid_search_params(self):
        self.cuda=  ptg.InputField("True", prompt="use cuda: ")
        self.model_name=  ptg.InputField("f.pt", prompt="model name: ")
        self.seq_len=  ptg.InputField("10", prompt="sequence length: ")
        self.lr=  ptg.InputField("1e-3", prompt="learning rate: ")
        self.batch_size=  ptg.InputField("1024", prompt="batch size: ")
        self.epochs= ptg.InputField("10", prompt="number of epochs: ")
        self.split= ptg.InputField("0.9", prompt="split fraction: ")
        self.log_name= ptg.InputField("Hospital_log.xes", prompt="log name: ")
        self.case_id_key= ptg.InputField("case:concept:name", prompt="case id key: ")
        self.case_activity_key= ptg.InputField("concept:name", prompt="activity key: ")
        self.case_timestamp_key= ptg.InputField("time:timestamp", prompt="timestamp key: ")

        self.hidden_dim_lower= ptg.InputField("100", prompt="hidden dim. lower bound: ")
        self.hidden_dim_upper= ptg.InputField("200", prompt="hidden dim. upper bound: ")
        self.hidden_dim_step= ptg.InputField("50", prompt="hidden dim. step: ")
        
        self.mlp_dim_lower= ptg.InputField("100", prompt="mlp dim. lower bound: ")
        self.mlp_dim_upper= ptg.InputField("200", prompt="mlp dim. upper bound: ")
        self.mlp_dim_step= ptg.InputField("100", prompt="mlp dim. step: ")


        self.emb_dim_lower= ptg.InputField("100", prompt="emb dim. lower bound: ")
        self.emb_dim_upper= ptg.InputField("200", prompt="emb dim. upper bound: ")
        self.emb_dim_step= ptg.InputField("100", prompt="emb dim. step: ")

        container = [
            ptg.Label(f"set parameters for grid search"),
            self.cuda , 
            self.model_name ,
            self.seq_len ,
            self.lr ,
            self.batch_size ,
            self.epochs ,
            self.split, 
            self.log_name,
            self.case_id_key, 
            self.case_activity_key, 
            self.case_timestamp_key,
            self.hidden_dim_lower, 
            self.hidden_dim_upper, 
            self.hidden_dim_step, 
            self.mlp_dim_lower, 
            self.mlp_dim_upper, 
            self.mlp_dim_step, 
            self.emb_dim_lower, 
            self.emb_dim_upper, 
            self.emb_dim_step,
            ptg.Button("continue", lambda *_: self.pp.switch_window(self.start_grid_search()))
        ]

        window = ptg.Window(*container)
        window.center()
        return window


    def trainer_main_menu(self) : 
        container = ptg.Container(
            "select one training alternative", 
            "", 
            ptg.Button("set params manually", lambda *_: self.pp.switch_window(self.set_training_params())), 
            "",
            ptg.Button("grid search", lambda *_: self.pp.switch_window(self.set_grid_search_params())), 
            "",
            ptg.Button("random search", lambda *_: self.pp.switch_window(self.set_random_search_params()))
        )

        window = ptg.Window(container, box="DOUBLE")
        window.center()
        return window