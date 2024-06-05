import ProcessProphet
import pytermgui as ptg
import json
import requests
import base64
from loggers import logger_set_params_cli
import ProcessProphetStart
from dotenv import load_dotenv
import os
from process_prophet_modes import ProcessProphetMode






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
        window = ptg.Window(*container, box="DOUBLE")
        window.center()
        self.pp.switch_window(window)
    

    def return_to_menu(self):
        """
        returns to p.p. start
        """
        pp_start = ProcessProphetStart.ProcessProphetStart(self.pp, False)


    @staticmethod
    def check_types_factory(all_params = True, is_random= False, is_grid= False):
        """
        this decorator ensures the correctness of the given inputs  
        and displays an error in case something is wrong
        """
        def check_types(func):
            def wrapper(self, *args, **kwargs):

                if self.cuda.value != 'True' and self.cuda.value != "False":
                    container= ptg.Container( 
                        ptg.Label(f"cuda should be True/False"),
                        ptg.Button("[black]back", lambda *_: self.pp.switch_window(self.trainer_main_menu()))
                    )
                    window = ptg.Window( container, box="DOUBLE")
                    window.center()
                    return window
                elif self.log_name.value not in os.listdir(self.pp.state.input_logs_path):
                    container= ptg.Container( 
                        ptg.Label(f"log does not exist"),
                        ptg.Label(f"{self.log_name.value}"),
                        ptg.Button("[black]back", lambda *_: self.pp.switch_window(self.trainer_main_menu()))
                    )
                    window = ptg.Window( container, box="DOUBLE")
                    window.center()
                    return window
                elif self.model_name.value in  os.listdir(self.pp.state.models_path):
                    container= ptg.Container( 
                        ptg.Label(f"model already exists"),
                        ptg.Button("[black]back", lambda *_: self.pp.switch_window(self.trainer_main_menu()))
                    )
                    window = ptg.Window( container, box="DOUBLE")
                    window.center()
                    return window
                elif not self.seq_len.value.isdigit():
                    container= ptg.Container( 
                        ptg.Label(f"seq len should be an integer"),
                        ptg.Button("[black]back", lambda *_: self.pp.switch_window(self.trainer_main_menu()))
                    )
                    window = ptg.Window( container, box="DOUBLE")
                    window.center()
                    return window
                elif all_params and not self.emb_dim.value.isdigit():
                    container= ptg.Container( 
                        ptg.Label(f"emb dim should be an integer"),
                        ptg.Button("[black]back", lambda *_: self.pp.switch_window(self.trainer_main_menu()))
                    )
                    window = ptg.Window( container, box="DOUBLE")
                    window.center()
                    return window
                elif all_params and not self.hid_dim.value.isdigit():
                    container= ptg.Container( 
                        ptg.Label(f"hid dim should be an integer"),
                        ptg.Button("[black]back", lambda *_: self.pp.switch_window(self.trainer_main_menu()))
                    )
                    window = ptg.Window( container, box="DOUBLE")
                    window.center()
                    return window
                elif all_params and not self.batch_size.value.isdigit():
                    container= ptg.Container( 
                        ptg.Label(f"batch size should be an integer"),
                        ptg.Button("[black]back", lambda *_: self.pp.switch_window(self.trainer_main_menu()))
                    )
                    window = ptg.Window( container, box="DOUBLE")
                    window.center()
                    return window
                elif all_params and not self.mlp_dim.value.isdigit():
                    container= ptg.Container( 
                        ptg.Label(f"mlp dim should be an integer"),
                        ptg.Button("[black]back", lambda *_: self.pp.switch_window(self.trainer_main_menu()))
                    )
                    window = ptg.Window( container, box="DOUBLE")
                    window.center()
                    return window

                elif not all_params and not self.emb_dim_lower.value.isdigit():
                    container= ptg.Container( 
                        ptg.Label(f"emb dim lower must be an integer"),
                        ptg.Button("[black]back", lambda *_: self.pp.switch_window(self.trainer_main_menu()))
                    )
                    window = ptg.Window( container, box="DOUBLE")
                    window.center()
                    return window

                elif not all_params and not self.emb_dim_upper.value.isdigit():
                    container= ptg.Container( 
                        ptg.Label(f"emb dim upper must be an integer"),
                        ptg.Button("[black]back", lambda *_: self.pp.switch_window(self.trainer_main_menu()))
                    )
                    window = ptg.Window( container, box="DOUBLE")
                    window.center()
                    return window
                elif not all_params and not self.mlp_dim_lower.value.isdigit():
                    container= ptg.Container( 
                        ptg.Label(f"mlp dim lower must be an integer"),
                        ptg.Button("[black]back", lambda *_: self.pp.switch_window(self.trainer_main_menu()))
                    )
                    window = ptg.Window( container, box="DOUBLE")
                    window.center()
                    return window

                elif not all_params and not self.mlp_dim_upper.value.isdigit():
                    container= ptg.Container( 
                        ptg.Label(f"mlp dim upper must be an integer"),
                        ptg.Button("[black]back", lambda *_: self.pp.switch_window(self.trainer_main_menu()))
                    )
                    window = ptg.Window( container, box="DOUBLE")
                    window.center()
                    return window
                elif not all_params and not self.hidden_dim_lower.value.isdigit():
                    container= ptg.Container( 
                        ptg.Label(f"hidden dim lower must be an integer"),
                        ptg.Button("[black]back", lambda *_: self.pp.switch_window(self.trainer_main_menu()))
                    )
                    window = ptg.Window( container, box="DOUBLE")
                    window.center()
                    return window

                elif not all_params and not self.hidden_dim_upper.value.isdigit():
                    container= ptg.Container( 
                        ptg.Label(f"hidden dim upper must be an integer"),
                        ptg.Button("[black]back", lambda *_: self.pp.switch_window(self.trainer_main_menu()))
                    )
                    window = ptg.Window( container, box="DOUBLE")
                    window.center()
                    return window

                elif is_grid and not self.emb_dim_step.value.isdigit():
                    container= ptg.Container( 
                        ptg.Label(f"emb dim step must be an integer"),
                        ptg.Button("[black]back", lambda *_: self.pp.switch_window(self.trainer_main_menu()))
                    )
                    window = ptg.Window( container, box="DOUBLE")
                    window.center()
                    return window
                elif is_grid and not self.mlp_dim_step.value.isdigit():
                    container= ptg.Container( 
                        ptg.Label(f"mlp dim step must be an integer"),
                        ptg.Button("[black]back", lambda *_: self.pp.switch_window(self.trainer_main_menu()))
                    )
                    window = ptg.Window( container, box="DOUBLE")
                    window.center()
                    return window
                elif is_grid and not self.hidden_dim_step.value.isdigit():
                    container= ptg.Container( 
                        ptg.Label(f"hidden dim step must be an integer"),
                        ptg.Button("[black]back", lambda *_: self.pp.switch_window(self.trainer_main_menu()))
                    )
                    window = ptg.Window( container, box="DOUBLE")
                    window.center()
                    return window


                elif is_random and not self.iterations.value.isdigit():
                    container= ptg.Container( 
                        ptg.Label(f"iterations should be an integer"),
                        ptg.Button("[black]back", lambda *_: self.pp.switch_window(self.trainer_main_menu()))
                    )
                    window = ptg.Window( container, box="DOUBLE")
                    window.center()
                    return window
                elif self.log_name.value[-3:]!="xes" and self.log_name.value[-3:] != "csv": 
                    container= ptg.Container( 
                        ptg.Label(f"only xes/csv supported"),
                        ptg.Button("[black]back", lambda *_: self.pp.switch_window(self.trainer_main_menu()))
                    )
                    window = ptg.Window( container, box="DOUBLE")
                    window.center()
                    return window
                try: 
                    float(self.lr.value)
                except ValueError: 
                    container= ptg.Container( 
                        ptg.Label(f"lr should be float"),
                        ptg.Button("[black]back", lambda *_: self.pp.switch_window(self.trainer_main_menu()))
                    )
                    window = ptg.Window( container, box="DOUBLE")
                    window.center()
                    return window

                try: 
                    val = float(self.split.value)
                    if not (val>0 and val<1): 
                        container= ptg.Container( 
                            ptg.Label(f"split must be in range (0,1)"),
                            ptg.Button("[black]back", lambda *_: self.pp.switch_window(self.trainer_main_menu()))
                        )
                        window = ptg.Window( container, box="DOUBLE")
                        window.center()
                        return window
                except ValueError: 
                    container= ptg.Container( 
                        ptg.Label(f"lr should be float"),
                        ptg.Button("[black]back", lambda *_: self.pp.switch_window(self.trainer_main_menu()))
                    )
                    window = ptg.Window( container, box="DOUBLE")
                    window.center()
                    return window

                return func(self,*args, **kwargs)
            return wrapper
        return check_types 

    @check_types_factory(all_params=True, is_grid=False, is_random=False)
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
        cuda= True if  self.cuda.value== "True"  else False
        
        params = {
            "path_to_log": f"{input_logs_path}/{self.log_name.value}" , 
            "split": self.split.value, 
            "model_path": f"{self.pp.state.models_path}/{self.model_name.value}", 
            "case_id": self.case_id_key.value, 
            "activity_key":  self.case_activity_key.value, 
            "timestamp_key":  self.case_timestamp_key.value, 
            "cuda": cuda, 
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
            data = response.json()

            statistics = data["training_statistics"]
            

            container =[  
                "training successful", 
                f"time error: {statistics['time error']}", 
                f"accuracy: {statistics['acc']}", 
                f"recall: {statistics['recall']}", 
                f"f1-score: {statistics['f1']}", 
                ptg.Button(f"{self.pp.button_color}training menu", lambda *_: self.pp.switch_window(self.trainer_main_menu())), 
                ptg.Button(f"{self.pp.button_color}action menu", lambda *_:  self.return_to_menu())
            ]
        else: 
            container = [ 
                "training FAILED:",
                ptg.Button("[black]back", lambda *_: self.pp.switch_window(self.set_training_params()))
            ]
        window = ptg.Window(*container, box="DOUBLE")
        window.center()
        return window





    def set_training_params(self): 
        self.cuda=  ptg.InputField("True", prompt="use cuda: ")

        self.model_name=  ptg.InputField("f.pt", prompt="model name: ")
        self.seq_len=  ptg.InputField("10", prompt="sequence length: ")
        self.emb_dim=  ptg.InputField("32", prompt="embedding dimension: ")
        self.hid_dim=  ptg.InputField("32", prompt="hidden dimension: ")
        self.mlp_dim=  ptg.InputField("16", prompt="mlp dimension: ")
        self.epochs=  ptg.InputField("10", prompt="number of epochs: ")
        self.batch_size=  ptg.InputField("1024", prompt="batch size: ")
        
        self.lr=  ptg.InputField("1e-3", prompt="learning rate: ")
        self.split=  ptg.InputField("0.9", prompt="split fraction: ")

        self.log_name=  ptg.InputField("Hospital_log.xes", prompt="log name: ")

        self.case_id_key=  ptg.InputField("case:concept:name", prompt="case id key: ")
        self.case_activity_key=  ptg.InputField("concept:name", prompt="activity key: ")
        self.case_timestamp_key=  ptg.InputField("time:timestamp", prompt="timestamp key: ")

        left_container = ptg.Container( 
            ptg.Label(f"set parameters for training"),
            SERVER_NAME,
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
            "",
            ptg.Button(f"{self.pp.button_color}continue", lambda *_: self.pp.switch_window(self.start_training())),
            "",
            ptg.Button("[black]back", lambda *_: self.pp.switch_window(self.trainer_main_menu()))
        )
        
        logs = [log for log in os.listdir(self.pp.state.input_logs_path)]
        logs = logs[:min(len(logs),4 )] #: to not overflow the terminal

        right_container = ptg.Container(
            f"[underline]First {len(logs)} logs in project:", *logs
        ).center()



        window = ptg.Window(ptg.Splitter(left_container, right_container), width = self.pp.window_width)
        #window = ptg.Window(*container)
        window.center()
        return window

    @check_types_factory(all_params=False, is_grid=True, is_random=False)
    def start_grid_search(self) : 
        self.loading("preprocessing data...")
        
        input_logs_path= self.pp.state.input_logs_path

        is_xes = True if self.log_name.value[-3:] == "xes"  else False

        cuda= True if  self.cuda.value== "True"  else False
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
            "cuda": cuda, 
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
            


            container =[
                "training successful", 
                f"accuracy: {accuracy}", 
                ptg.Button(f"{self.pp.button_color}training menu", lambda *_: self.pp.switch_window(self.trainer_main_menu())), 
                ptg.Button(f"{self.pp.button_color}action menu", lambda *_:  self.return_to_menu())
            ]
        else: 
            container =[  
                "training FAILED:", 
                ptg.Button(f"{self.pp.button_color}back", lambda *_: self.pp.switch_window(self.set_grid_search_params()))
            ]
        window = ptg.Window(*container, box="DOUBLE")
        window.center()
        return window


    @check_types_factory(all_params=False, is_grid=False, is_random=True)
    def start_random_search(self) : 
        self.loading("preprocessing data...")
        
        input_logs_path= self.pp.state.input_logs_path

        is_xes = True if self.log_name.value[-3:] == "xes"  else False

        cuda= True if  self.cuda.value== "True"  else False
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
            "cuda": cuda, 
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

            container = [ 
                "training successful", 
                f"accuracy: {accuracy}", 
                ptg.Button(f"{self.pp.button_color}training menu", lambda *_: self.pp.switch_window(self.trainer_main_menu())), 
                ptg.Button(f"{self.pp.button_color}action menu", lambda *_:  self.return_to_menu())
            ]
        else: 
            container = [ 
                "training FAILED:", 
                ptg.Button(f"{self.pp.button_color}back", lambda *_: self.pp.switch_window(self.set_random_search_params()))
            ]
        window = ptg.Window(*container, box="DOUBLE")
        window.center()
        return window

    def set_random_search_params(self):

        if self.pp.mode == ProcessProphetMode.advanced:
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

            container = ptg.Container( 
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
                self.iterations,"",
                ptg.Button(f"{self.pp.button_color}continue", lambda *_: self.pp.switch_window(self.start_random_search())),"",
                ptg.Button("[black]back", lambda *_: self.pp.switch_window(self.trainer_main_menu()))
            )
            
            logs = [log for log in os.listdir(self.pp.state.input_logs_path)]
            logs = logs[:min(len(logs),4 )] #: to not overflow the terminal

            right_container = ptg.Container(
                f"[underline]First {len(logs)} logs in project:", *logs
            ).center()
            window = ptg.Window(ptg.Splitter(container, right_container), width = self.pp.window_width)
            #window = ptg.Window(*container)
            window.center()
            return window
        elif self.pp.mode == ProcessProphetMode.quick:
            self.cuda=  ptg.InputField("False", prompt="use cuda: ") #: no cuda assumed

            self.lr=  ptg.InputField("1e-3", prompt="learning rate: ")  #: set to 1e-3 by default, this is a usual value

            self.epochs= ptg.InputField("30", prompt="number of epochs: ") #: set to 30 by default
            self.split= ptg.InputField("0.9", prompt="split fraction: ") #: set to 0.9 by default
            


            self.batch_size=  ptg.InputField("1024", prompt="batch size: ")
            self.model_name=  ptg.InputField("f.pt", prompt="model name: ")
            self.seq_len=  ptg.InputField("10", prompt="sequence length: ") 
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

            container = ptg.Container( 
                ptg.Label(f"set parameters for grid search"),
                self.model_name ,
                self.seq_len ,
                self.batch_size ,
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
                self.iterations,"",
                ptg.Button(f"{self.pp.button_color}continue", lambda *_: self.pp.switch_window(self.start_random_search())),"",
                ptg.Button("[black]back", lambda *_: self.pp.switch_window(self.trainer_main_menu()))
            )
            
            logs = [log for log in os.listdir(self.pp.state.input_logs_path)]
            logs = logs[:min(len(logs),4 )] #: to not overflow the terminal

            right_container = ptg.Container(
                f"[underline]First {len(logs)} logs in project:", *logs
            ).center()
            window = ptg.Window(ptg.Splitter(container, right_container), width = self.pp.window_width)
            #window = ptg.Window(*container)
            window.center()
            return window


    def set_grid_search_params(self):
        if self.pp.mode == ProcessProphetMode.advanced:
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

            container = ptg.Container( 
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
                ptg.Button(f"{self.pp.button_color}continue", lambda *_: self.pp.switch_window(self.start_grid_search())),
                ptg.Button("[black]back", lambda *_: self.pp.switch_window(self.trainer_main_menu()))
            ).center()

            
            logs = [log for log in os.listdir(self.pp.state.input_logs_path)]
            logs = logs[:min(len(logs),4 )] #: to not overflow the terminal

            right_container = ptg.Container(
                f"[underline]First {len(logs)} logs in project:", *logs
            ).center()
            window = ptg.Window(ptg.Splitter(container, right_container), width = self.pp.window_width)
            #window = ptg.Window(*container)
            window.center()
            return window
        elif self.pp.mode  == ProcessProphetMode.quick: 
            self.cuda=  ptg.InputField("True", prompt="use cuda: ")
            self.model_name=  ptg.InputField("f.pt", prompt="model name: ")
            self.seq_len=  ptg.InputField("30", prompt="sequence length: ")
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

            container = ptg.Container( 
                ptg.Label(f"set parameters for grid search"),
                self.model_name ,
                self.seq_len ,
                self.batch_size ,
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
                ptg.Button(f"{self.pp.button_color}continue", lambda *_: self.pp.switch_window(self.start_grid_search())),
                ptg.Button("[black]back", lambda *_: self.pp.switch_window(self.trainer_main_menu()))
            ).center()

            
            logs = [log for log in os.listdir(self.pp.state.input_logs_path)]
            logs = logs[:min(len(logs),4 )] #: to not overflow the terminal

            right_container = ptg.Container(
                f"[underline]First {len(logs)} logs in project:", *logs
            ).center()
            window = ptg.Window(ptg.Splitter(container, right_container), width = self.pp.window_width)
            #window = ptg.Window(*container)
            window.center()
            return window


    def trainer_main_menu(self) : 
        if self.pp.mode == ProcessProphetMode.advanced: 
            container = ptg.Container(
                "select one training alternative", 
                "", 
                ptg.Button(f"{self.pp.button_color}set params manually", lambda *_: self.pp.switch_window(self.set_training_params())), 
                "",
                ptg.Button(f"{self.pp.button_color}grid search", lambda *_: self.pp.switch_window(self.set_grid_search_params())), 
                "",
                ptg.Button(f"{self.pp.button_color}random search", lambda *_: self.pp.switch_window(self.set_random_search_params())),
                "",
                ptg.Button("[black]back", lambda *_: self.return_to_menu())
            )

            window = ptg.Window(*container, box="DOUBLE")
            window.center()

            return window

        elif self.pp.mode == ProcessProphetMode.quick: 
            container = ptg.Container(
                "select one training alternative", 
                "",
                ptg.Button(f"{self.pp.button_color}grid search", lambda *_: self.pp.switch_window(self.set_grid_search_params())), 
                "",
                ptg.Button(f"{self.pp.button_color}random search", lambda *_: self.pp.switch_window(self.set_random_search_params())),
                "",
                ptg.Button("[black]back", lambda *_: self.return_to_menu())
            )

            window = ptg.Window(*container, box="DOUBLE")
            window.center()

            return window