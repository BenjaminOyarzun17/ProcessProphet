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

    def do_import(self, is_xes): 
        """
        call the api
        TODO: create API endpoint for importing and exporting the generated
        event_df as csv.
        """
        self.loading("preprocessing data...")
        params = {
            "is_xes": is_xes, 
            "case_id":self.case_id_key.value, 
            "activity_key": self.case_activity_key.value, 
            "timestamp_key": self.case_timestamp_key.value, 
            "path_to_log": f"{self.pp.state.input_logs_path}/{self.log_name.value}",
            "export_path": f"{self.pp.state.decoded_dfs_path}/{self.export_name.value }",
        }
        response = requests.get(
            "http://localhost:5000/import_log", 
            params = params,
            timeout =6000
        )
        if response.status_code == 200: 
            container = ptg.Container(
                "preprocessing successful", 
                ptg.Button("preprocessing menu", lambda *_: self.pp.switch_window(self.preprocessing_main_menu())), 
                ptg.Button("action menu", lambda *_:  self.return_to_menu())
            )
        else: 
            container = ptg.Container(
                "preprocessing FAILED:"
            )
    
        window = ptg.Window(container, box="DOUBLE")
        window.center()
        return window


    def import_log(self, is_xes): 
        logs_labels = []
        logs =  os.listdir(f"{self.pp.state.input_logs_path}")
        for log in logs:
            logs_labels.append(ptg.Label(f"- {log}"))
        self.log_name=  ptg.InputField(logs[0], prompt="event log name: ")
        self.case_id_key = ptg.InputField("case:concept:name" if is_xes else "", prompt="case id column name: ")
        self.case_activity_key =  ptg.InputField("concept:name" if is_xes else "", prompt="activity column name: ")
        self.case_timestamp_key = ptg.InputField("time:timestamp" if is_xes else "", prompt="timestamp column name: ")
        self.export_name= ptg.InputField("", prompt="export name: ")
        self.sep= ptg.InputField(",", prompt="csv column separator: ")

        contents = [
            ptg.Label(f"set parameters for importing"),
            "", 
            "available logs in project: ",
            ptg.Container(*logs_labels),
            self.log_name, 
            "",
            self.case_id_key,
            self.case_activity_key,
            self.case_timestamp_key,
            self.export_name, 
            
        ]
        if not is_xes: 
            contents.append(self.sep)
        contents.append(ptg.Button("continue", lambda *_: self.pp.switch_window(self.do_import(is_xes))))

        container= ptg.Container(
            *contents
        ).center()
        window = ptg.Window(container, box="DOUBLE")
        window.center()
        return window


    def preprocessing_main_menu(self) : 
        container = ptg.Container(
            "select one action:", 
            ptg.Button("import csv", lambda *_: self.pp.switch_window(self.import_log(False))), 
            "",
            ptg.Button("import xes", lambda *_: self.pp.switch_window(self.import_log(True))),
            "",
            ptg.Button("filter df", lambda *_: self.pp.switch_window())
        )

        window = ptg.Window(container, box="DOUBLE")
        window.center()
        return window