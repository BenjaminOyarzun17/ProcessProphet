import ProcessProphet
import pytermgui as ptg
import json
import requests
import base64
from loggers import logger_set_params_cli
import ProcessProphetStart

class ProcessProphetTrain: 
    def __init__(self, pp):
        self.pp = pp
        self.pp.switch_window(self.trainer_main_menu())

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



    def start_training(self) : 
        self.loading("preprocessing data...")
        input_logs_path= self.pp.state.input_logs_path

        is_xes = True if self.log_name.value[-3:] == "xes"  else False
        
        params = {
            "path_to_log": f"{input_logs_path}/{self.log_name.value}" , 
            "split": self.split.value, 
            "model_path": f"{self.pp.state.models_path}/{self.model_name.value}", 
            "case_id": self.case_id_key.value, 
            "activity_key":  self.case_activity_key.value, 
            "timestamp_key":  self.case_timestamp_key.value, 
            "cuda": self.cuda.value, 
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
            "http://localhost:5000/train_nn", 
            json= params,
            timeout =6000
        )
        if response.status_code == 200: 
            logger_set_params_cli.debug(response.content)
            data = response.json()

            statistics = data["training_statistics"]
            config = data["config"]
            with open(f"{self.pp.state.models_path}/{self.model_name.value}.config.json", "w") as f:
                json.dump(config,f)
            
            encoded_file_content = data.get('file_content')

            file_content = base64.b64decode(encoded_file_content)
            with open(f"{self.pp.state.models_path}/{self.model_name.value}", 'wb') as file:
                file.write(file_content)

            container = ptg.Container(
                "training successful", 
                f"accuracy: {statistics["acc"]}", 
                f"recall: {statistics["recall"]}", 
                f"f1-score: {statistics["f1"]}", 
                ptg.Button("training menu", lambda *_: self.pp.switch_window(self.trainer_main_menu())), 
                ptg.Button("action menu", lambda *_:  self.return_to_menu())
            )
        else: 
            container = ptg.Container(
                "training FAILED:"
            )
        window = ptg.Window(container, box="DOUBLE")
        window.center()
        return window


    def set_training_params(self): 
        self.cuda=  ptg.InputField("True", prompt="use cuda: ")
        self.model_name=  ptg.InputField("f.pt", prompt="model name: ")
        self.seq_len=  ptg.InputField("10", prompt="sequence length: ")
        self.emb_dim=  ptg.InputField("32", prompt="embedding dimension: ")
        self.hid_dim=  ptg.InputField("32", prompt="hidden dimension: ")
        self.mlp_dim=  ptg.InputField("16", prompt="mlp dimension: ")
        self.lr=  ptg.InputField("1e-3", prompt="learning rate: ")
        self.batch_size=  ptg.InputField("1024", prompt="batch size: ")
        self.epochs=  ptg.InputField("10", prompt="number of epochs: ")
        self.split=  ptg.InputField("0.9", prompt="split fraction: ")
        self.log_name=  ptg.InputField("BPI_Challenge_2019.xes", prompt="log name: ")
        self.case_id_key=  ptg.InputField("case:concept:name", prompt="case id key: ")
        self.case_activity_key=  ptg.InputField("concept:name", prompt="activity key: ")
        self.case_timestamp_key=  ptg.InputField("time:timestamp", prompt="timestamp key: ")
        


        container = [
            ptg.Label(f"set parameters for training"),
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
            ptg.Button("continue", lambda *_: self.pp.switch_window(self.start_training()))
        ]

        window = ptg.Window(*container)
        window.center()
        return window


    def trainer_main_menu(self) : 
        container = ptg.Container(
            ptg.Button("set params manually", lambda *_: self.pp.switch_window(self.set_training_params())), 
            ptg.Button("grid search", lambda *_: self.pp.switch_window(self.set_training_params())), 
            ptg.Button("random search", lambda *_: self.pp.switch_window(self.set_training_params()))
        )

        window = ptg.Window(container, box="DOUBLE")
        window.center()
        return window