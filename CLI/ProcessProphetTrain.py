import ProcessProphet
import pytermgui as ptg
import json
import requests
from loggers import logger_set_params_cli
import ProcessProphetStart
from dotenv import load_dotenv
import os
from process_prophet_modes import ProcessProphetMode






load_dotenv()
SERVER_NAME= os.getenv('SERVER_NAME')
SERVER_PORT= os.getenv('SERVER_PORT')

class ProcessProphetTrain: 
    """
    this class provides three basic functions: 
    - train RNN by setting params manually 
    - train RNN using grid search
    - train RNN using random search
    each one of these options generates a `.pt` file containing the pytorch model and a 
    `.config.json` file containing the RNN training configuration, encoders, and other data
    relevant to process prophet. 
    """
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
        returns to p.p. start. Start is set to False, since we dont want to select the project again. 
        this makes sense for example, when the user wants to make predictions after having trained the RNN.
        """
        pp_start = ProcessProphetStart.ProcessProphetStart(self.pp, start = False)



    def start_training(self) : 
        """
        carries out a training request. 

        on successful request completion the following side effects/ outputs are expected: 
        :return model: a model `pt` file is saved in the models folder 
        :return config: a models `config.json` information for the server is saved in the models folder
        as a json file. 

        the training statistics (time error, accuracy, recall, f1 score)  are 
        displayed on screen. 

        if the training is unsuccessful, the error returned by the server is displayed on the CLI. 
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
            data = response.json()
            error = data["error"]
            container = [ 
                "training FAILED:",
                "",
                f"{error}", 
                "",
                ptg.Button("[black]back", lambda *_: self.pp.switch_window(self.set_training_params()))
            ]
        window = ptg.Window(*container, box="DOUBLE")
        window.center()
        return window





    def set_training_params(self):
        """
        user can either start the training with the displayed default parameters or alternatively adapt the parameters to their
        own preference

        side effect:
        -the modified parameters are stored in a container and then the training function is called
        -parameters are displayed in the window
        -second window to display the logs contained in this project as a visual aid
        """ 
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

        #: contains the form for setting the parameters
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

        #: contains a list of the logs contained in the input logs path.
        right_container = ptg.Container(
            f"[underline]First {len(logs)} logs in project:", *logs
        ).center()



        window = ptg.Window(ptg.Splitter(left_container, right_container), width = self.pp.window_width)
        #window = ptg.Window(*container)
        window.center()
        return window

    def start_grid_search(self):
        """
        sends a request to the server with all the needed parameters to carry out grid search training
        and in case of a successful computation of the request by the server the accuracy of the trained
        model is displayed in a new window. It is then possible to return to the action (manager selection) or training menu. 

        if the request fails because e.g. it exceeds the timeout of 6000 the error is displayed in a new window and 
        the user can go back to the window where the parameters are displayed
        """ 
        self.loading("preprocessing data...")
        
        input_logs_path= self.pp.state.input_logs_path

        #: checks the if the file extension is xes. 
        is_xes = True if self.log_name.value[-3:] == "xes"  else False

        #: casting bool("False") also returns True 
        cuda= True if  self.cuda.value== "True"  else False

        #: search params for grid search
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
            data = response.json()

            accuracy = data["acc"]
            #: display accuracy on success
            container =[
                "training successful", 
                f"accuracy: {accuracy}", 
                ptg.Button(f"{self.pp.button_color}training menu", lambda *_: self.pp.switch_window(self.trainer_main_menu())), 
                ptg.Button(f"{self.pp.button_color}action menu", lambda *_:  self.return_to_menu())
            ]
        else: 
            #: display error on fail
            data =response.json()
            error = data[""]
            container =[  
                "training FAILED:", 
                "",
                f"{error}", 
                "",
                ptg.Button(f"{self.pp.button_color}back", lambda *_: self.pp.switch_window(self.set_grid_search_params()))
            ]
        window = ptg.Window(*container, box="DOUBLE")
        window.center()
        return window


    def start_random_search(self) :
        """
        sends a request to the server with all the needed parameters to do a random search training
        and in case of a successful computation of the request by the server the accuracy of the trained
        model is displayed in a new window. It is then possible to return to the action or training menu

        if the request fails because e.g. it exceeds the timeout of 6000 the error is displayed in a new window and 
        the user can go back to the window where the parameters are displayed
        """ 
        self.loading("preprocessing data...") #: shows the loading screen until the response es received. 
        
        input_logs_path= self.pp.state.input_logs_path

        is_xes = True if self.log_name.value[-3:] == "xes"  else False

        cuda= True if  self.cuda.value== "True"  else False

        #: params for random search. here we only need lower and upper bounds
        sp = {
            "hidden_dim":[self.hidden_dim_lower.value,self.hidden_dim_upper.value] ,
            "mlp_dim":[self.mlp_dim_lower.value, self.mlp_dim_upper.value] ,
            "emb_dim":[self.emb_dim_lower.value, self.emb_dim_upper.value] 
        } 

        #: note the iterations param needed for random search.
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

            #: display accuracy on success
            container = [ 
                "training successful", 
                f"accuracy: {accuracy}", 
                ptg.Button(f"{self.pp.button_color}training menu", lambda *_: self.pp.switch_window(self.trainer_main_menu())), 
                ptg.Button(f"{self.pp.button_color}action menu", lambda *_:  self.return_to_menu())
            ]
        else: 
            data = response.json()
            error = data["error"]
            container = [ 
                "training FAILED:", 
                "",
                f"{error}", 
                "",
                ptg.Button(f"{self.pp.button_color}back", lambda *_: self.pp.switch_window(self.set_random_search_params()))
            ]
        window = ptg.Window(*container, box="DOUBLE")
        window.center()
        return window

    def set_random_search_params(self):
        """
        used to set the parameters for random search training alternative

        this function distinguishes between quick mode and advanced mode by giving more options to customize
        the hyperparameters in the advanced mode whereas in the base mode only the most important parameters
        can be modified by the user

        Side effects:
        -initializes window with default parameters where th user can adjust them
        -initializes window where all the event logs of the current project are listed that can be used
        for the training
        -random search can be called if the user confirms the indicated parameters
        """
        if self.pp.mode == ProcessProphetMode.advanced:
            #: show all params in case of advanced mode
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

            #: shows the available logs.
            right_container = ptg.Container(
                f"[underline]First {len(logs)} logs in project:", *logs
            ).center()
            window = ptg.Window(ptg.Splitter(container, right_container), width = self.pp.window_width)
            #window = ptg.Window(*container)
            window.center()
            return window
        elif self.pp.mode == ProcessProphetMode.quick:
            #: show some params in case of quick mode
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
        """
        used to set the parameters for grid search training alternative

        this function distinguishes between quick mode and advanced mode by giving more options to customize
        the hyperparameters in the advanced mode whereas in the base mode only the most important parameters
        can be modified by the user

        Side effects:
        -initializes window with default parameters where th user can adjust them
        -initializes window where all the event logs of the current project are listed that can be used
        for the training
        -grid search can be called if the user confirms the indicated parameters
        """
        if self.pp.mode == ProcessProphetMode.advanced:
            #: show all params in case of advanced mode
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
            #: show some params in case of advanced mode
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
        """
        this function displays the main menuy for the trainer manager. 

        depending on the mode the current project is running in, the user can choose a training alternative
        and will be redirected to a new window where the parameters for the chosen alternative are displayed.

        it is also possible to return to the previous menu.
        """ 
        if self.pp.mode == ProcessProphetMode.advanced: 
            #: the set params manually option is only available in the advanced mode
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
            #: only grid search and random search are available in this mode.
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