"""
This modules allows prediction generation.
"""
import os
import requests
import pytermgui as ptg
import ProcessProphetStart
from process_prophet_modes import ProcessProphetMode
from dotenv import load_dotenv

load_dotenv()
SERVER_NAME= os.getenv('SERVER_NAME')
SERVER_PORT= os.getenv('SERVER_PORT')
TIMEOUT= int(os.getenv('TIMEOUT'))

class ProcessProphetPredict: 
    def __init__(self, pp):
        """
        initialize ProcessProphet instance and prediction main menu
        other state parameters that make sense in the context of prediction generation might also be saved 
        here

        :param pp: the ProcessProphet instance in charge of window management 
        """
        self.pp = pp
        self.pp.switch_window(self.prediction_main_menu())

    def prediction_main_menu(self) :
        """
        menu that returns the window of the selected prediction
        """ 
        container = ptg.Container(
            "select a prediction generation method", 
            "", 
            ptg.Button(f"{self.pp.button_color}single prediction", lambda *_: self.pp.switch_window(self.set_single_prediction_params())), 
            "",
            ptg.Button(f"{self.pp.button_color}multiple prediction", lambda *_: self.pp.switch_window(self.set_multiple_prediction_params())), 
            "",
            ptg.Button(f"{self.pp.button_color}back", lambda *_: self.return_to_menu()), 
        )

        window = ptg.Window(container, box="DOUBLE")
        window.center()
        return window
    
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


    def get_single_prediction(self) : 
        """
        carries out a single prediction request. 

        side effects/ outputs: 
        marker, timestamp and the probability of the single prediction are displayed
        """
        self.loading("predicting next event...")
        input_logs_path= self.pp.state.partial_traces_path
        # checks the if the file extension is xes
        is_xes = True if self.log_name.value[-3:] == "xes"  else False
        
        #: parameters that are passed to the server for further computations
        params = {
            "path_to_log": f"{input_logs_path}/{self.log_name.value}" ,  
            "path_to_model": f"{self.pp.state.models_path}/{self.model_name.value}", 
            "case_id": self.case_id_key.value, 
            "activity_key":  self.case_activity_key.value, 
            "timestamp_key":  self.case_timestamp_key.value,  
            "is_xes": is_xes,
            "config": f"{self.pp.state.models_path}/{self.model_name.value[:-3]}.config.json",
        }


        response = requests.post(
            f"http://{SERVER_NAME}:{SERVER_PORT}/single_prediction", 
            json= params,
            timeout =TIMEOUT
        )
        if response.status_code == 200: 
            data = response.json()

            statistics = data
            
            #: container to display the computed event, time and its probability
            container = ptg.Container(
                "single prediction successful", 
                f"predicted time: {statistics['predicted_time']}", 
                f"predicted event: {statistics['predicted_event']}",
                f"probability: {statistics['probability']}%",
                "",
                ptg.Button(f"{self.pp.button_color}back", lambda *_:self.pp.switch_window(self.prediction_main_menu())), 
                "",
                ptg.Button(f"{self.pp.button_color}return to menu", lambda *_:self.return_to_menu()), 
            )
        else: 
            data = response.json()
            error = data["error"]
            #: container to display that an error occured in the request
            container = ptg.Container(
                "single prediction FAILED:",
                "",
                f"{error}", 
                "",
                ptg.Button(f"{self.pp.button_color}back", lambda *_: self.pp.switch_window(self.prediction_main_menu()))
            )
        window = ptg.Window(container, box="DOUBLE")
        window.center()
        return window
    
    def set_single_prediction_params(self):
        """
        user can modify the given parameters for a single prediction which are then stored in the container and also displayed in the current window

        user can also start the prediction with the continue button or return to the previous menu with the back button
        """


        if self.pp.mode == ProcessProphetMode.advanced:
            self.model_name=  ptg.InputField("f.pt", prompt="model name: ")
            self.log_name=  ptg.InputField("partial_input.csv", prompt="log name: ")
            self.case_id_key=  ptg.InputField("case:concept:name", prompt="case id key: ")
            self.case_activity_key=  ptg.InputField("concept:name", prompt="activity key: ")
            self.case_timestamp_key=  ptg.InputField("time:timestamp", prompt="timestamp key: ")

            container = [
                ptg.Label(f"set parameters for prediction"), 
                self.model_name ,
                self.log_name,
                self.case_id_key, 
                self.case_activity_key, 
                self.case_timestamp_key,
                ptg.Button(f"{self.pp.button_color}continue", lambda *_: self.pp.switch_window(self.get_single_prediction())),
                ptg.Button(f"{self.pp.button_color}back", lambda *_: self.pp.switch_window(self.prediction_main_menu()))
                ]
            # display the container with the parameters
        window = ptg.Window(*container)
        window.center()
        return window
    
    def get_multiple_prediction(self) : 
        """
        carries out a multiple prediction request. 

        side effects/ outputs: 
        markers and timestamps of the multiple prediction are displayed in a seperate file
        """
        self.loading("predicting next event...")
        input_logs_path= self.pp.state.partial_traces_path
        #: checks the if the file extension is xes
        is_xes = True if self.log_name.value[-3:] == "xes"  else False
        #: parameters that are passed to the server for further computations
        params = {
            "path_to_log": f"{input_logs_path}/{self.log_name.value}" ,  
            "path_to_model": f"{self.pp.state.models_path}/{self.model_name.value}", 
            "case_id": self.case_id_key.value, 
            "activity_key":  self.case_activity_key.value, 
            "timestamp_key":  self.case_timestamp_key.value,  
            "is_xes": is_xes,
            "config": f"{self.pp.state.models_path}/{self.model_name.value[:-3]}.config.json",
            "depth": self.depth.value,
            "degree": self.degree.value, 
            "prediction_file_name": f"{self.pp.state.multiple_predictions_path}/{self.prediction_file_name_input.value}"
        }

        response = requests.post(
            f"http://{SERVER_NAME}:{SERVER_PORT}/multiple_prediction", 
            json= params,
            timeout =8000
        )
        if response.status_code == 200: 
            data = response.json()

            paths = data
            #: container that indicates success of the request and shows the filename where the predictions are stored 
            container = ptg.Container(
                f"Multiple predictions stored in {params['prediction_file_name']}", 
                ptg.Button(f"{self.pp.button_color}back", lambda *_: self.pp.switch_window(self.prediction_main_menu()))
            )
        else: 
            data = response.json()
            error = data["error"]
            #: container that indicates an error that occured from the request
            container = ptg.Container(
                "multiple prediction FAILED:",
                "",
                f"{error}", 
                "",
                ptg.Button(f"{self.pp.button_color}back", lambda *_: self.pp.switch_window(self.prediction_main_menu()))
            )
        window = ptg.Window(container, box="DOUBLE")
        window.center()
        return window
    
    def set_multiple_prediction_params(self):
        """
        function to display the default values for a multiple prediction and grants the user access to 
        modify the given parameters for multiple predictions which are then stored in the container and also displayed in the current window

        user can also start the prediction with the continue button or return to the previous menu with the back button
        """
        self.model_name=  ptg.InputField("f.pt", prompt="model name: ")
        self.log_name=  ptg.InputField("partial_input.csv", prompt="log name: ")
        self.case_id_key=  ptg.InputField("case:concept:name", prompt="case id key: ")
        self.case_activity_key=  ptg.InputField("concept:name", prompt="activity key: ")
        self.case_timestamp_key=  ptg.InputField("time:timestamp", prompt="timestamp key: ")
        self.depth= ptg.InputField("5", prompt="depth: ") #: amount of following events that should be predicted
        self.degree= ptg.InputField("3", prompt="degree: ") #: pick the k most likely ones.
        self.prediction_file_name_input = ptg.InputField("mp1.json", prompt= "predictions file name: ")

        container = [
            ptg.Label(f"set parameters for prediction"), 
            self.model_name ,
            self.log_name,
            self.prediction_file_name_input,
            self.case_id_key, 
            self.case_activity_key, 
            self.case_timestamp_key,
            self.depth,
            self.degree,
            ptg.Button(f"{self.pp.button_color}continue", lambda *_: self.pp.switch_window(self.get_multiple_prediction())), 
            ptg.Button(f"{self.pp.button_color}back", lambda *_: self.pp.switch_window(self.prediction_main_menu()))
            ]
        #: display the container with the parameters
        window = ptg.Window(*container)
        window.center()
        return window
    
    def return_to_menu(self):
        """
        returns to p.p. start
        """
        pp_start = ProcessProphetStart.ProcessProphetStart(self.pp, False)