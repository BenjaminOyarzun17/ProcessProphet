from ProcessProphet import ProcessProphet
import pytermgui as ptg


class ProcessProphetTrain: 
    def __init__(self, pp):
        self.pp = pp
        self.pp.switch_window(self.trainer_main_menu())


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


    def set_training_params(self): 
        self.cuda=  ptg.InputField("", prompt="use cuda: ")
        self.model_name=  ptg.InputField("", prompt="model name: ")
        self.seq_len=  ptg.InputField("10", prompt="sequence length: ")
        self.emb_dim=  ptg.InputField("32", prompt="embedding dimension: ")
        self.hid_dim=  ptg.InputField("32", prompt="hidden dimension: ")
        self.mlp_dim=  ptg.InputField("16", prompt="mlp dimension: ")
        self.lr=  ptg.InputField("1e-3", prompt="learning rate: ")
        self.batch_size=  ptg.InputField("1024", prompt="batch size: ")
        self.epochs=  ptg.InputField("10", prompt="number of epochs: ")
        self.split=  ptg.InputField("0.9", prompt="split fraction: ")

        container = ptg.Container(
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
            ptg.Button("continue", lambda *_: self.pp.switch_window(self.start_training()))
        ).center()

        window = ptg.Window(container, box="DOUBLE")
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