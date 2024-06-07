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


class ProcessProphetModel:
    def __init__(self, pp):
        self.pp = pp
        self.pp.switch_window(self.model_main_menu())

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
    
    def model_main_menu(self):
        container = ptg.Container(
            "Select one action", 
            "", 
            ptg.Button("Create a predictive event log", lambda *_: self.pp.switch_window(self.set_predictive_log())), 
            "", 
            ptg.Button("Run process mining", lambda *_: self.pp.switch_window(self.prediction_log())), 
            "", 
            ptg.Button("Run conformance checking", lambda *_: self.pp.switch_window(self.prediction_log())), 
            "",
            ptg.Button("Back", lambda *_: self.pp.switch_window(self.return_to_menu())), 
        )

        window = ptg.Window(container, box="DOUBLE")
        window.center()
        return window 

    def get_predictive_log(self): 

        self.loading("preprocessing data...")

        is_xes = True if self.log_name.value[-3:] == "xes"  else False
        
        params = {
            "path_to_model":f"{self.pp.state.models_path}/{self.model_name.value}",
            "path_to_log":f"{self.pp.state.input_logs_path}/{self.log_name.value}" ,
            "case_id":self.case_id_key.value,
            "activity_key":self.case_activity_key.value,
            "timestamp_key":self.case_timestamp_key.value,
            "new_log_path":f"{self.pp.state.predictive_logs_path}/{self.predictive_event_log_name.value}",
            "non_stop":self.non_stop.value,
            "upper":self.upper .value,
            "random_cuts":self.random_cuts.value,
            "cut_length":self.cut_length.value,
            "config":f"{self.pp.state.models_path}/{self.model_name.value[:-3]}.config.json",
            "is_xes": is_xes, 
            "sep": "," 
        }  

        response = requests.post(
            f"http://{SERVER_NAME}:{SERVER_PORT}/generate_predictive_log", 
            json= params,
            timeout =6000
        )
        if response.status_code == 200: 
            data = response.json()

           
            

            container =[  
                "predictive process model generated successfully", 
                ptg.Button(f"{self.pp.button_color}back", lambda *_: self.pp.switch_window(self.model_main_menu())), 
                ptg.Button(f"{self.pp.button_color}action menu", lambda *_:  self.return_to_menu())
            ]
        else: 
            #data = response.json()
            #error = data["error"]
            container = [ 
                "training FAILED:",
                #"",
                #f"{error}", 
                #"",
                ptg.Button("[black]back", lambda *_: self.pp.switch_window(self.set_training_params()))
            ]
        window = ptg.Window(*container, box="DOUBLE")
        window.center()
        return window


    def set_predictive_log(self) :
        self.model_name=  ptg.InputField("f.pt", prompt="model name: ")
        self.log_name=  ptg.InputField("Hospital_log.xes", prompt="log name: ")
        self.case_id_key=  ptg.InputField("case:concept:name", prompt="case id key: ")
        self.case_activity_key=  ptg.InputField("concept:name", prompt="activity key: ")
        self.case_timestamp_key=  ptg.InputField("time:timestamp", prompt="timestamp key: ")
        self.predictive_event_log_name  = ptg.InputField("predicitive_log1.csv", prompt= "predictive log name: ")
        self.non_stop = ptg.InputField("True", prompt="run until end event: ")
        self.upper = ptg.InputField("30", prompt="non stop upper bound: ")
        self.random_cuts = ptg.InputField("True", prompt="use random cuts: ")
        self.cut_length = ptg.InputField("0", prompt="cut length: ")
        container = [
            "Enter the following params:",
            self.model_name,
            self.log_name,
            self.case_id_key,
            self.case_activity_key,
            self.case_timestamp_key,
            self.predictive_event_log_name,
            self.non_stop,
            self.upper,
            self.random_cuts,
            self.cut_length,
            ptg.Button("continue",lambda *_: self.pp.switch_window(self.get_predictive_log()) ),
            ptg.Button("back",lambda *_: self.pp.switch_window(self.model_main_menu()) )
        ]
        window = ptg.Window(*container, width = self.pp.window_width)
        #window = ptg.Window(*container)
        window.center()
        return window
        


