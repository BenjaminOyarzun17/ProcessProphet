from ProcessProphet import ProcessProphet
import pytermgui as ptg
import os
import requests

import ProcessProphetStart

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

   
    def preprocessing_main_menu(self) : 
        container = ptg.Container(
            "select one action:", 
            ptg.Button("filter df", lambda *_: self.pp.switch_window())
        )

        window = ptg.Window(container, box="DOUBLE")
        window.center()
        return window