import pytermgui as ptg
from dataclasses import dataclass
import os
from loggers import logger_set_params_cli

@dataclass
class PPStateData: 
    projects_path: str
    current_project: str|None
    model_trained: bool
    predictive_df_generated: bool
    petri_net_generated: bool


class ProcessProphet:
    def __init__(self):
        self.manager = ptg.WindowManager()
        # Initially add the main menu window
        self.state = PPStateData("projects", None, False, False, False) 
        self.current_window = None

    def set_current_window(self, window): 
        self.current_window = window

    def remove_current_window(self): 
        self.manager.remove(self.current_window)

    def switch_window(self, new_window):
        # Remove the current window
        if self.current_window !=None:
            self.remove_current_window()
        logger_set_params_cli.debug(self.current_window)
        # Add the new window

        self.set_current_window(new_window)
        self.manager.add(new_window)
        self.manager.focus(new_window)

    def run(self):
        with self.manager:
            self.manager.run()

    
