from ProcessProphet import ProcessProphet
import pytermgui as ptg


class ProcessProphetPreprocessing: 
    def __init__(self, pp):
        self.pp = pp
        self.pp.switch_window(self.preprocessing_main_menu())


    def start_training(self) : 
        params = {
            "is_xes": None, 
            "path_to_log":None , 
            "sep": None, 
            "case_id": None, 
            "activity_key": None, 
            "timestamp_key": None, 
            "split": None, 
            "cuda": self.cuda.value, 
            "model_name": self.model_name.value
        } 


    def do_import(self): 
        """
        call the api
        TODO: create API endpoint for importing and exporting the generated
        event_df as csv.
        """
    def import_csv(self): 
        self.log_name=  ptg.InputField("", prompt="event log name: ")
        self.case_id_key = ptg.InputField("", prompt="case id column name: ")
        self.case_activity_key =  ptg.InputField("", prompt="activity column name: ")
        self.case_timestamp_key = ptg.InputField("", prompt="timestamp column name: ")
        self.sep= ptg.InputField(",", prompt="separator for the csv columns: ")
        container = ptg.Container(
            ptg.Label(f"set parameters for importing"),
            self.cuda , 
            self.model_name ,
            self.seq_len ,
            self.emb_dim ,
            self.hid_dim ,
            self.mlp_dim ,
            self.lr ,
            self.batch_size ,
            self.epochs ,
            ptg.Button("continue", lambda *_: self.pp.switch_window(self.do_import()))
        ).center()

        window = ptg.Window(container, box="DOUBLE")
        window.center()
        return window


    def preprocessing_main_menu(self) : 
        container = ptg.Container(
            ptg.Button("import csv", lambda *_: self.pp.switch_window()), 
            ptg.Button("import xes", lambda *_: self.pp.switch_window()),
            ptg.Button("filter df", lambda *_: self.pp.switch_window())
        )

        window = ptg.Window(container, box="DOUBLE")
        window.center()
        return window