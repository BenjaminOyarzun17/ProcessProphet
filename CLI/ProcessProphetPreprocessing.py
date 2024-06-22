"""
This modules gives access to some preprocessing functions.
"""
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
        Initializes a ProcessProphetPreprocessing object and sets up the preprocessing main menu.

        Args:
            pp (ProcessProphet): The ProcessProphet instance in charge of window management.
        """
        self.pp = pp  # Reference to the ProcessProphet object
        self.pp.switch_window(self.preprocessing_main_menu())  # Starts with the preprocessing main menu
    

    #: this decorator is used for type checking
    @staticmethod
    def check_types(func):
        """
        Decorator that checks if the file is an accepted file type (xes/csv) and if the file exists in the project directory.

        Args:
            func: The function that `check_types` decorates.

        Returns:
            The decorated function.

        Side effects:
            If the file type or existence restrictions are not followed, a new window with the corresponding error is indicated.
        """
        def wrapper(self, *args, **kwargs):
            """
            first it checks if the file is an excepted file type 
            (xes/csv) and the file also exists in the directory of 
            the project and then calls the original function
            """
            if self.log_name.value[-3:]!="xes" and self.log_name.value[-3:] != "csv":
                # error if file is in the wrong file type 
                container= ptg.Container( 
                    ptg.Label(f"only xes/csv supported"),
                    ptg.Button(f"{self.pp.button_color}back", lambda *_: self.pp.switch_window(self.preprocessing_main_menu()))
                )
                window = ptg.Window( container, box="DOUBLE")
                window.center()
                return window
            elif self.log_name.value not in os.listdir(self.pp.state.input_logs_path):
                # error if file does not exist in the directory
                container= ptg.Container( 
                    ptg.Label(f"the log does not exist"),
                    ptg.Button(f"{self.pp.button_color}back", lambda *_: self.pp.switch_window(self.preprocessing_main_menu()))
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
        Indicates all the parameters needed to add unique start and end activities to each trace.

        The user can modify these parameters in the left side of the window.

        The function also displays the first few Log file names in the current project on the right side of the window.

        Side effects:
            - Initializes a window with default parameters where the user can adjust them.
            - Initializes a window where all the event logs of the current project are listed for preprocessing.
            - Calls the `add_unique_start_end` function if the user confirms the indicated parameters.
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
            ptg.Button(f"{self.pp.button_color}back", lambda *_: self.pp.switch_window(self.preprocessing_main_menu()))
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
        and the user can modify them in the left side of the window.

        The function also indicates the first few Log file names in the current project on
        the right side of the window.

        Side effects:
            - Initializes a window with default parameters where the user can adjust them.
            - Initializes a window where all the event logs of the current project are listed that can be used
            for the preprocessing.
            - Calls the `remove_duplicates` function if the user confirms the indicated parameters.
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
            ptg.Button(f"{self.pp.button_color}back", lambda *_: self.pp.switch_window(self.preprocessing_main_menu()))
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
        and the user can modify them in the left side of the window.

        The function also indicates the first few Log file names in the current project on
        the right side of the window.

        Side effects:
            - Initializes a window with default parameters where the user can adjust them.
            - Initializes a window where all the event logs of the current project are listed that can be used
            for the preprocessing.
            - Calls the `replace_nan_with_mode` function if the user confirms the indicated parameters.
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
            ptg.Button(f"{self.pp.button_color}back", lambda *_: self.pp.switch_window(self.preprocessing_main_menu()))
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
        Displays the main menu for the preprocessing manager.

        The user can choose one of the three alternatives:
            - Replacing NaN values in the log.
            - Removing duplicate rows in the log.
            - Adding unique start and end activities to each trace.

        It is also possible to return to the previous menu.
        """
        replace = f"{self.pp.button_color}replace NaN in activity column with mode"
        remove= f"{self.pp.button_color}remove duplicate rows"
        add= f"{self.pp.button_color}add unique start and end activities"

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
