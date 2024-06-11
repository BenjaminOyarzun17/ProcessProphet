from ProcessProphet import ProcessProphet
import pytermgui as ptg
import os
import requests

import ProcessProphetStart
from dotenv import load_dotenv


load_dotenv()
SERVER_NAME= os.getenv('SERVER_NAME')
SERVER_PORT= os.getenv('SERVER_PORT')
TIMEOUT= int(os.getenv('TIMEOUT'))


class ProcessProphetPreprocessing: 
    def __init__(self, pp):
        """
        initialize ProcessProphet Object and Preprocessing main menu

        :param pp: the ProcessProphet instance in charge of window management 
        """
        self.pp = pp #: reference to the ProcessProphet object
        self.pp.switch_window(self.preprocessing_main_menu()) #: starts with the preprocessing main menu



    #: this decorator is used for type checking
    @staticmethod
    def check_types(func):
        """
        decorator that can be called to check if the file is an
        excepted file type (xes/csv) and the file also exists in
        the directory of the project

        :func: the function that check_types decorates

        side effect:
        if the restrictions are not followed a new window with the
        corresponding error is indicated
        """
        def wrapper(self, *args, **kwargs):
            """
            first it checks if the file is an excepted file type 
            (xes/csv) and the file also exists in the directory of 
            the project and then calls the original function

            :*args:, :**kwargs: parameters to store an unspecified 
            number auf arguments and keyword arguments because we dont
            necessarily know how many arguments the function we call the
            decorator on needs
            """
            if self.log_name.value[-3:]!="xes" and self.log_name.value[-3:] != "csv":
                # error if file is in the wrong file type 
                container= ptg.Container( 
                    ptg.Label(f"only xes/csv supported"),
                    ptg.Button("[black]back", lambda *_: self.pp.switch_window(self.preprocessing_main_menu()))
                )
                window = ptg.Window( container, box="DOUBLE")
                window.center()
                return window
            elif self.log_name.value not in os.listdir(self.pp.state.input_logs_path):
                # error if file does not exist in the directory
                container= ptg.Container( 
                    ptg.Label(f"the log does not exist"),
                    ptg.Button("[black]back", lambda *_: self.pp.switch_window(self.preprocessing_main_menu()))
                )
                window = ptg.Window( container, box="DOUBLE")
                window.center()
                return window
            # call original function the decorator was called on
            return func(self,*args, **kwargs)
        return wrapper
   
    def loading(self, message = ""):
        """
        function to indicate a message in a new window e.g. to 
        show that a process is loading
        """ 
        container = ptg.Container(
            "Loading...", 
            message
        )
        window = ptg.Window(container, box="DOUBLE")
        window.center()
        self.pp.switch_window(window)

    def return_to_menu(self):
        #: returns to the manager selection menu of `ProcessProphetStart`. Therefore, `start` is set to `False`
        pp_start = ProcessProphetStart.ProcessProphetStart(self.pp, start = False)




    @check_types
    def handle_replace_nan_with_mode(self):
        """
        first decorator is used to ensure the file can be preprocessed

        sends a request to the server with all the needed parameters to do replace all NaN values in the log
        and in case of a successful computation of the request by the server the path where the preprocessed log
        is stored in will be indicated in a new window

        if the request fails because e.g. it exceeds the timeout of TIMEOUT the error is displayed in a new window and 
        the user can go back to the window where the parameters are displayed
        """ 
        self.loading("preprocessing data...") # loading screen while data is being preprocessed
        input_logs_path= self.pp.state.input_logs_path
        
        #: checks if extension is xes. otherwise csv assumed
        is_xes = True if self.log_name.value[-3:] == "xes"  else False
        
        params = {
            "path_to_log": f"{input_logs_path}/{self.log_name.value}" , 
            "case_id": self.case_id_key.value, 
            "activity_key":  self.case_activity_key.value, 
            "timestamp_key":  self.case_timestamp_key.value, 
            "is_xes": is_xes, 
            "save_path": f"{input_logs_path}/{self.save_path.value}" ,
            "sep": ","
        } 

        response = requests.post(
            f"http://{SERVER_NAME}:{SERVER_PORT}/replace_with_mode", 
            json= params,
            timeout =TIMEOUT
        )
        if response.status_code==200:
            # succesful and indicate path for preprocessed data
            data = response.json()
            container= ptg.Container( 
                ptg.Label(f"success"),
                f"log saved in path {data['save_path']}"
                "",
                ptg.Button(f"{self.pp.button_color}back", lambda *_: self.pp.switch_window(self.preprocessing_main_menu())),
            )
        else:
            # error ocurred 
            data = response.json()
            container= ptg.Container( 
                ptg.Label(f"error: {data['error']}"),
                "",
                ptg.Button(f"{self.pp.button_color}back", lambda *_: self.pp.switch_window(self.preprocessing_main_menu())),
            )

        window = ptg.Window( container, box="DOUBLE")
        window.center()
        return window

    @check_types
    def handle_remove_duplicates(self):
        """
        first decorator is used to ensure the file can be preprocessed

        sends a request to the server with all the needed parameters to do remove duplicate rows from the log
        and in case of a successful computation of the request by the server the path where the preprocessed log
        is stored in will be indicated in a new window

        if the request fails because e.g. it exceeds the timeout of TIMEOUT the error is displayed in a new window and 
        the user can go back to the window where the parameters are displayed
        """
        self.loading("preprocessing data...") # loading screen while data is being preprocessed
        input_logs_path= self.pp.state.input_logs_path
        
        is_xes = True if self.log_name.value[-3:] == "xes"  else False
        
        params = {
            "path_to_log": f"{input_logs_path}/{self.log_name.value}" , 
            "case_id": self.case_id_key.value, 
            "activity_key":  self.case_activity_key.value, 
            "timestamp_key":  self.case_timestamp_key.value, 
            "is_xes": is_xes, 
            "save_path": f"{input_logs_path}/{self.save_path.value}" ,
            "sep": ","
        } 

        response = requests.post(
            f"http://{SERVER_NAME}:{SERVER_PORT}/remove_duplicates", 
            json= params,
            timeout =TIMEOUT
        )
        if response.status_code==200:
            # successful and indicate path for preprocessed data
            data = response.json()
            container= ptg.Container( 
                ptg.Label(f"success"),
                f"log saved in path {data['save_path']}"
                "",
                ptg.Button(f"{self.pp.button_color}back", lambda *_: self.pp.switch_window(self.preprocessing_main_menu())),
            )
        else:
            # error ocurred 
            data = response.json()
            container= ptg.Container( 
                ptg.Label(f"error: {data['error']}"),
                "",
                ptg.Button(f"{self.pp.button_color}back", lambda *_: self.pp.switch_window(self.preprocessing_main_menu())),
            )

        window = ptg.Window( container, box="DOUBLE")
        window.center()
        return window
    

 


    @check_types
    def handle_add_unique_start_end(self):
        """
        first decorator is used to ensure the file can be preprocessed

        sends a request to the server with all the needed parameters to do add unique start end end activities to each trace
        and in case of a successful computation of the request by the server the path where the preprocessed log
        is stored in will be indicated in a new window

        if the request fails because e.g. it exceeds the timeout of TIMEOUT the error is displayed in a new window and 
        the user can go back to the window where the parameters are displayed
        """
        self.loading("preprocessing data...") # loading screen while data is being preprocessed
        input_logs_path= self.pp.state.input_logs_path
        is_xes = True if self.log_name.value[-3:] == "xes"  else False
        
        params = {
            "path_to_log": f"{input_logs_path}/{self.log_name.value}" , 
            "case_id": self.case_id_key.value, 
            "activity_key":  self.case_activity_key.value, 
            "timestamp_key":  self.case_timestamp_key.value, 
            "is_xes": is_xes, 
            "save_path": f"{input_logs_path}/{self.save_path.value}" ,
            "sep": ","
        } 

        response = requests.post(
            f"http://{SERVER_NAME}:{SERVER_PORT}/add_unique_start_end", 
            json= params,
            timeout =TIMEOUT
        )
        if response.status_code==200:
            # successful request
            data = response.json()
            container= ptg.Container( 
                ptg.Label(f"success"),
                f"log saved in path {data['save_path']}"
                "",
                ptg.Button(f"{self.pp.button_color}back", lambda *_: self.pp.switch_window(self.preprocessing_main_menu())),
            )
        else: 
            # an error occured
            data = response.json()
            container= ptg.Container( 
                ptg.Label(f"error: {data['error']}"),
                "",
                ptg.Button(f"{self.pp.button_color}back", lambda *_: self.pp.switch_window(self.preprocessing_main_menu())),
            )

        window = ptg.Window( container, box="DOUBLE")
        window.center()
        return window


    def add_unique_start_end(self):
        """
        This function indicates all the parameters that are needed to add unique start
        and end activities to each trace and the user can modify them in the left side of the window

        The function also indicates the first few Log file names in the current project on
        the right side of the window

        Side effects:
        -initializes window with default parameters where the user can adjust them
        -initializes window where all the event logs of the current project are listed that can be used
        for the preprocessing
        -add_unique_start_end can be called if the user confirms the indicated parameters
        """ 
        self.log_name=  ptg.InputField("Hospital_log.xes", prompt="log name: ")
        self.case_id_key=  ptg.InputField("case:concept:name", prompt="case id key: ")
        self.case_activity_key=  ptg.InputField("concept:name", prompt="activity key: ")
        self.case_timestamp_key=  ptg.InputField("time:timestamp", prompt="timestamp key: ")
        self.save_path=  ptg.InputField("HL_unique_start.csv", prompt="output log name:") # Name of the preprocessed copy of the log

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
        """
        This function indicates all the parameters that are needed to remove duplicate rows
        and the user can modify them in the left side of the window

        The function also indicates the first few Log file names in the current project on
        the right side of the window

        Side effects:
        -initializes window with default parameters where the user can adjust them
        -initializes window where all the event logs of the current project are listed that can be used
        for the preprocessing
        -remove_duplicates can be called if the user confirms the indicated parameters
        """ 
        self.log_name=  ptg.InputField("Hospital_log.xes", prompt="log name: ")
        self.case_id_key=  ptg.InputField("case:concept:name", prompt="case id key: ")
        self.case_activity_key=  ptg.InputField("concept:name", prompt="activity key: ")
        self.case_timestamp_key=  ptg.InputField("time:timestamp", prompt="timestamp key: ")
        self.save_path=  ptg.InputField("HL_no_dup.csv", prompt="output log name:") # Name of the preprocessed copy of the log

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



    def replace_nan_with_mode(self):
        """
        This function indicates all the parameters that are needed to replace NaN values
        and the user can modify them in the left side of the window

        The function also indicates the first few Log file names in the current project on
        the right side of the window

        Side effects:
        -initializes window with default parameters where the user can adjust them
        -initializes window where all the event logs of the current project are listed that can be used
        for the preprocessing
        -replace_nan_with_mode can be called if the user confirms the indicated parameters
        """ 
        self.log_name=  ptg.InputField("Hospital_log.xes", prompt="log name: ")
        self.case_id_key=  ptg.InputField("case:concept:name", prompt="case id key: ")
        self.case_activity_key=  ptg.InputField("concept:name", prompt="activity key: ")
        self.case_timestamp_key=  ptg.InputField("time:timestamp", prompt="timestamp key: ")
        self.save_path=  ptg.InputField("HL_nan_to_mode.csv", prompt="output log name:") # Name of the preprocessed copy of the log
        #indicates params
        left_container = ptg.Container( 
            ptg.Label(f"enter relevant information"),
            self.log_name,
            self.case_id_key, 
            self.case_activity_key, 
            self.case_timestamp_key,
            self.save_path,
            "",
            ptg.Button(f"{self.pp.button_color}continue", lambda *_: self.pp.switch_window(self.handle_replace_nan_with_mode())),
            "",
            ptg.Button("[black]back", lambda *_: self.pp.switch_window(self.preprocessing_main_menu()))
        )
        
        logs = [log for log in os.listdir(self.pp.state.input_logs_path)]
        logs = logs[:min(len(logs),4 )] #: to not overflow the terminal
        #shows logs in the current project
        right_container = ptg.Container(
            f"[underline]First {len(logs)} logs in project:", *logs
        ).center()

        window = ptg.Window(ptg.Splitter(left_container, right_container), width = self.pp.window_width)
        window.center()
        return window




    def preprocessing_main_menu(self):
        """
        this function displays the main menu for the preprocessing manager. 

        depending on how the data should be preprocessed the user can choose
        one of the three alternatives
            1. replacing NaN values in the log
            2. removing duplicate rows in the log
            3. adding unique start and end activities to each trace

        it is also possible to return to the previous menu.
        """
        replace = "replace NaN in activity column with mode"
        remove= "remove duplicate rows"
        add= "add unique start and end activities"

        container = ptg.Container(
            "select one action:", 
            ptg.Button(label = replace,onclick= lambda *_: self.pp.switch_window(self.replace_nan_with_mode())),
            "",
            ptg.Button(label = remove,onclick= lambda *_: self.pp.switch_window(self.remove_duplicates())),
            "",
            ptg.Button(label = add, onclick=lambda *_: self.pp.switch_window(self.add_unique_start_end())), 
            "",
            ptg.Button("back", lambda *_: self.return_to_menu())  

            
        )

        window = ptg.Window(container, box="DOUBLE", width= self.pp.window_width)
        window.center()
        return window
