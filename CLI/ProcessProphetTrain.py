from ProcessProphet import ProcessProphet
import pytermgui as ptg


class ProcessProphetTrain: 
    def __init__(self, pp):
        self.pp = pp
        self.pp.switch_window(self.trainer_main_menu())
    

    def set_training_params(self): 
        cuda=  ptg.InputField("", prompt="use cuda: ")
        model_name=  ptg.InputField("", prompt="model name: ")
        seq_len=  ptg.InputField("10", prompt="sequence length: ")
        emb_dim=  ptg.InputField("32", prompt="embedding dimension: ")
        hid_dim=  ptg.InputField("32", prompt="hidden dimension: ")
        mlp_dim=  ptg.InputField("16", prompt="mlp dimension: ")
        lr=  ptg.InputField("1e-3", prompt="learning rate: ")
        batch_size=  ptg.InputField("1024", prompt="batch size: ")
        epochs=  ptg.InputField("10", prompt="number of epochs: ")



        container = ptg.Container(
            ptg.Label(f"set parameters for training"),
            cuda, 
            model_name, 
            seq_len, 
            emb_dim, 
            hid_dim, 
            mlp_dim, 
            lr, 
            batch_size, 
            epochs
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