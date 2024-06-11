import pytermgui as ptg
from dataclasses import dataclass
import os
from process_prophet_modes import ProcessProphetMode


with open("CLI/styles.yaml", "r") as styles_file: 
    styles_content = styles_file.read()

CONFIG = styles_content





@dataclass
class PPStateData: 
    """
    data class used for state conservation. 
    """
    projects_path: str # path where all projects are saved

    current_project: str|None # current project name

    model_trained: bool # whether a model has been trained

    predictive_df_generated: bool # same
    petri_net_generated: bool # same


    #: the following paths are subfolders of the current project. they are set for convenience. 
    input_logs_path: str|None
    models_path: str|None
    petri_nets_path: str|None
    predictive_logs_path:str|None
    partial_traces_path: str|None
    multiple_predictions_path: str|None
    decoded_dfs_path:str|None
    mode: ProcessProphetMode | None



class SingletonMeta(type):
    """
    singleton metaclass taken from https://refactoring.guru/design-patterns/singleton/python/example
    """

    _instances = {}

    def __call__(cls, *args, **kwargs):
        if cls not in cls._instances:
            instance = super().__call__(*args, **kwargs)
            cls._instances[cls] = instance
        return cls._instances[cls]



class ProcessProphet(metaclass = SingletonMeta):
    """
    this class is intended for window management. 
    this class works as a singleton. the other ProcessProphet classes (such as ProcessProphetStart)
    will be always provided with the same instance of this class and will basically determine 
    the content of `self.current_window`. 

    there can only be one instance of this class, as there is only one terminal to draw in. therefore this 
    class is a singleton.
    """
    def __init__(self):
        self.state = PPStateData("projects", None, False, False, False, None, None, None,None,None,None, None, None) 
        
        #: window manager object from pytermgui. this object handles 
        # window lifecycle.  windows have nice properties such as
        # being resizable. 
        self.manager = ptg.WindowManager() 
        
        #: the current window's content
        self.current_window = None 

        #: variable used for styling
        self.button_color = "[black]"

        #: use 80% of the window width
        self.window_width = int(os.get_terminal_size(0)[0]*0.8)
        self.window_height = 50



    def set_current_window(self, window): 
        """
        sets the current window
        :param window: window object, the new window.
        """
        self.current_window = window 
    def remove_current_window(self): 
        """
        removes the current window
        :param window: window object, the new window.
        """
        self.manager.remove(self.current_window)

    def switch_window(self, new_window):
        """
        in charge of switching windows.  
        :param new_window: the new window. 
        """
        with ptg.YamlLoader() as loader: #: loads the styles from `styles.yaml`
            loader.load(CONFIG)
        if self.current_window !=None: #for not initialized case
            self.remove_current_window()

        #: changes the window
        self.set_current_window(new_window)
        self.manager.add(new_window)
        self.manager.focus(new_window)

    def run(self):
        with self.manager:
            #: run the app.
            self.manager.run()

    
