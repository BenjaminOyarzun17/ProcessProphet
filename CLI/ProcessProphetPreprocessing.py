from ProcessProphet import ProcessProphet
import pytermgui as ptg
import os
import requests

import ProcessProphetStart
from dotenv import load_dotenv


load_dotenv()
SERVER_NAME= os.getenv('SERVER_NAME')
SERVER_PORT= os.getenv('SERVER_PORT')



class ProcessProphetPreprocessing: 
    def __init__(self, pp):
        self.pp = pp
        self.pp.switch_window(self.preprocessing_main_menu())

   
    def loading(self, message = ""): 
        container = ptg.Container(
            "Loading...", 
            message
        )
        window = ptg.Window(container, box="DOUBLE")
        window.center()
        self.pp.switch_window(window)

    def return_to_menu(self):
        pp_start = ProcessProphetStart.ProcessProphetStart(self.pp)




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


    def handle_replace_nan_with_median(self):
        
        self.loading("preprocessing data...")
        input_logs_path= self.pp.state.input_logs_path
        
        is_xes = True if self.log_name.value[-3:] == "xes"  else False
        
        params = {
            "path_to_log": f"{input_logs_path}/{self.log_name.value}" , 
            "case_id": self.case_id_key.value, 
            "activity_key":  self.case_activity_key.value, 
            "timestamp_key":  self.case_timestamp_key.value, 
            "is_xes": is_xes, 
            "save_path": f"{input_logs_path}/{self.save_path.value}" 
        } 

        response = requests.post(
            f"http://{SERVER_NAME}:{SERVER_PORT}/replace_with_median", 
            json= params,
            timeout =6000
        )
        if response.status_code==200:
            data = response.json()
            container= ptg.Container( 
                ptg.Label(f"success"),
                f"log saved in path {data["save_path"]}"
                "",
                ptg.Button(f"{self.pp.button_color}back", lambda *_: self.pp.switch_window(self.preprocessing_main_menu())),
            )
        else: 
            container= ptg.Container( 
                ptg.Label(f"something went wrong..."),
                "",
                ptg.Button(f"{self.pp.button_color}back", lambda *_: self.pp.switch_window(self.preprocessing_main_menu())),
            )

        window = ptg.Window( container, box="DOUBLE")
        window.center()
        return window

    def handle_remove_duplicates(self):

        self.loading("preprocessing data...")
        input_logs_path= self.pp.state.input_logs_path
        
        is_xes = True if self.log_name.value[-3:] == "xes"  else False
        
        params = {
            "path_to_log": f"{input_logs_path}/{self.log_name.value}" , 
            "case_id": self.case_id_key.value, 
            "activity_key":  self.case_activity_key.value, 
            "timestamp_key":  self.case_timestamp_key.value, 
            "is_xes": is_xes, 
            "save_path": f"{input_logs_path}/{self.save_path.value}" 
        } 

        response = requests.post(
            f"http://{SERVER_NAME}:{SERVER_PORT}/remove_duplicates", 
            json= params,
            timeout =6000
        )
        if response.status_code==200:
            data = response.json()
            container= ptg.Container( 
                ptg.Label(f"success"),
                f"log saved in path {data["save_path"]}"
                "",
                ptg.Button(f"{self.pp.button_color}back", lambda *_: self.pp.switch_window(self.preprocessing_main_menu())),
            )
        else: 
            container= ptg.Container( 
                ptg.Label(f"something went wrong..."),
                "",
                ptg.Button(f"{self.pp.button_color}back", lambda *_: self.pp.switch_window(self.preprocessing_main_menu())),
            )

        window = ptg.Window( container, box="DOUBLE")
        window.center()
        return window
    


    def handle_add_unique_start_end(self):

        self.loading("preprocessing data...")
        input_logs_path= self.pp.state.input_logs_path
        
        is_xes = True if self.log_name.value[-3:] == "xes"  else False
        
        params = {
            "path_to_log": f"{input_logs_path}/{self.log_name.value}" , 
            "case_id": self.case_id_key.value, 
            "activity_key":  self.case_activity_key.value, 
            "timestamp_key":  self.case_timestamp_key.value, 
            "is_xes": is_xes, 
            "save_path": f"{input_logs_path}/{self.save_path.value}" 
        } 

        response = requests.post(
            f"http://{SERVER_NAME}:{SERVER_PORT}/add_unique_start_end", 
            json= params,
            timeout =6000
        )
        if response.status_code==200:
            data = response.json()
            container= ptg.Container( 
                ptg.Label(f"success"),
                f"log saved in path {data["save_path"]}"
                "",
                ptg.Button(f"{self.pp.button_color}back", lambda *_: self.pp.switch_window(self.preprocessing_main_menu())),
            )
        else: 
            container= ptg.Container( 
                ptg.Label(f"log already has the requested properties!"),
                "",
                ptg.Button(f"{self.pp.button_color}back", lambda *_: self.pp.switch_window(self.preprocessing_main_menu())),
            )

        window = ptg.Window( container, box="DOUBLE")
        window.center()
        return window



    def add_unique_start_end(self): 
        self.log_name=  ptg.InputField("Hospital_log.xes", prompt="log name: ")
        self.case_id_key=  ptg.InputField("case:concept:name", prompt="case id key: ")
        self.case_activity_key=  ptg.InputField("concept:name", prompt="activity key: ")
        self.case_timestamp_key=  ptg.InputField("time:timestamp", prompt="timestamp key: ")
        self.save_path=  ptg.InputField("HL_nan_to_medi.xes", prompt="output log name:")

        left_container = ptg.Container( 
            ptg.Label(f"enter relevant information"),
            self.log_name,
            self.case_id_key, 
            self.case_activity_key, 
            self.case_timestamp_key,
            self.save_path,
            "",
            ptg.Button(f"{self.pp.button_color}continue", lambda *_: self.pp.switch_window(self.handle_add_unique_start_end())),
            "",
            ptg.Button("[black]back", lambda *_: self.pp.switch_window(self.preprocessing_main_menu()))
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




    def remove_duplicates(self): 
        self.log_name=  ptg.InputField("Hospital_log.xes", prompt="log name: ")
        self.case_id_key=  ptg.InputField("case:concept:name", prompt="case id key: ")
        self.case_activity_key=  ptg.InputField("concept:name", prompt="activity key: ")
        self.case_timestamp_key=  ptg.InputField("time:timestamp", prompt="timestamp key: ")
        self.save_path=  ptg.InputField("HL_nan_to_medi.xes", prompt="output log name:")

        left_container = ptg.Container( 
            ptg.Label(f"enter relevant information"),
            self.log_name,
            self.case_id_key, 
            self.case_activity_key, 
            self.case_timestamp_key,
            self.save_path,
            "",
            ptg.Button(f"{self.pp.button_color}continue", lambda *_: self.pp.switch_window(self.handle_remove_duplicates())),
            "",
            ptg.Button("[black]back", lambda *_: self.pp.switch_window(self.preprocessing_main_menu()))
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



    def replace_nan_with_median(self): 
        self.log_name=  ptg.InputField("Hospital_log.xes", prompt="log name: ")
        self.case_id_key=  ptg.InputField("case:concept:name", prompt="case id key: ")
        self.case_activity_key=  ptg.InputField("concept:name", prompt="activity key: ")
        self.case_timestamp_key=  ptg.InputField("time:timestamp", prompt="timestamp key: ")
        self.save_path=  ptg.InputField("HL_nan_to_medi.xes", prompt="output log name:")

        left_container = ptg.Container( 
            ptg.Label(f"enter relevant information"),
            self.log_name,
            self.case_id_key, 
            self.case_activity_key, 
            self.case_timestamp_key,
            self.save_path,
            "",
            ptg.Button(f"{self.pp.button_color}continue", lambda *_: self.pp.switch_window(self.handle_replace_nan_with_median())),
            "",
            ptg.Button("[black]back", lambda *_: self.pp.switch_window(self.preprocessing_main_menu()))
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




    def preprocessing_main_menu(self) : 
        replace = "replace NaN in activity column with median"
        remove= "remove duplicate rows"
        add= "add unique start and end activities"

        container = ptg.Container(
            "select one action:", 
            ptg.Button(label = replace,onclick= lambda *_: self.pp.switch_window(self.replace_nan_with_median())),
            "",
            ptg.Button(label = remove,onclick= lambda *_: self.pp.switch_window(self.remove_duplicates())),
            "",
            ptg.Button(label = add, onclick=lambda *_: self.pp.switch_window(self.add_unique_start_end())), 
            "",
            ptg.Button("return to menu", lambda *_: self.return_to_menu())  

            
        )

        window = ptg.Window(container, box="DOUBLE", width= self.pp.window_width)
        window.center()
        return window