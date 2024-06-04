import pytermgui as ptg
from dataclasses import dataclass
import os
from loggers import logger_set_params_cli
from process_prophet_modes import ProcessProphetMode


with open("CLI/styles.yaml", "r") as styles_file: 
    styles_content = styles_file.read()

CONFIG = styles_content





@dataclass
class PPStateData: 
    """
    data class used for state conservation
    """
    projects_path: str # path where all projects are saved
    current_project: str|None # current project name
    model_trained: bool # whether a model has been trained
    predictive_df_generated: bool # same
    petri_net_generated: bool # same
    input_logs_path: str|None
    models_path: str|None
    petri_nets_path: str|None
    predictive_logs_path:str|None
    partial_traces_path: str|None
    multiple_predictions_path: str|None
    decoded_dfs_path:str|None
    mode: ProcessProphetMode | None




class ProcessProphet:
    """
    this class is intended for window management
    """
    def __init__(self):
        self.state = PPStateData("projects", None, False, False, False, None, None, None,None,None,None, None, None) 
        self.manager = ptg.WindowManager() #: manager object for the window
        self.current_window = None #: current window content

        self.button_color = "[black]"
        self.window_width = int(os.get_terminal_size(0)[0]*0.8)
        self.window_height = 50

    def set_current_window(self, window): 
        self.current_window = window #: setter for current window

    def remove_current_window(self): 
        self.manager.remove(self.current_window) #: removes the window's content

    def switch_window(self, new_window):
        # Remove the current window

        with ptg.YamlLoader() as loader:
            loader.load(CONFIG)
        if self.current_window !=None: #for not initialized case
            self.remove_current_window()



        #: set the widgets in the window.
        self.set_current_window(new_window)
        self.manager.add(new_window)
        self.manager.focus(new_window)

    def run(self):
        with self.manager:
            #: run the app.
            self.manager.run()

    
