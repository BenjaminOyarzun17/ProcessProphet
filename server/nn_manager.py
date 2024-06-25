"""
This module is in charge of training the NN Model and also testing it. 

The generated model may be exported and also imported later on by this class.

it supports manual trainig, random search and grid search. 
"""
import pandas as pd
from server import RMTPP_torch
from server import exceptions
from server import time_precision

from torch.utils.data import DataLoader
from tqdm import tqdm
import numpy as np
import torch
from collections import Counter
import random
from sklearn.preprocessing import LabelEncoder

from enum import Enum, auto


class Config: 
    """
    class containing the configuration for the model. 

    Attributes:
        seq_len (int): The sequence length used for the sliding window. 
        emb_dim (int): The embedding dimension.
        hid_dim (int): The hidden dimension.
        mlp_dim (int): The MLP dimension used for the LSTM.
        batch_size (int): The batch size.
        alpha (float): The alpha value.
        dropout (float): The dropout value.
        time_precision (TimePrecision): The time precision. Only NS is supported.
        lr (float): The learning rate.
        epochs (int): The number of epochs.
        importance_weight (str): The importance weight. (set to a default value as in the RMTPP implementation)
        verbose_step (int): The verbose step, just for logging purposes.
        cuda (bool): Whether to use the GPU.
        absolute_frequency_distribution (Counter): The absolute frequency distribution of the classes.
        case_id_le (LabelEncoder): The case ID label encoder.
        activity_le (LabelEncoder): The activity label encoder.
        exponent (int): The exponent used for the time conversion (see the preprocessing module).
        number_classes (int): The number of possible activities in the data.
        case_activity_key (str): The case activity key.
        case_timestamp_key (str): The case timestamp key.
        case_id_key (str): The case ID key.
    """
    def __init__(self):
        self.seq_len: int= 10
        self.emb_dim: int= 1500
        self.hid_dim:int=1500
        self.mlp_dim:int= 1500
        self.batch_size:int = 1024
        self.alpha: float= 0.05
        self.dropout:float= 0.1
        self.time_precision:time_precision.TimePrecision = time_precision.TimePrecision.NS
        self.lr:float = 1e-3
        self.epochs: int = 3 
        self.importance_weight: str = "store_true"
        self.verbose_step: int = 350
        self.cuda: bool = False 
        self.absolute_frequency_distribution:Counter = Counter()
        self.case_id_le:LabelEncoder = None
        self.activity_le:LabelEncoder = None
        self.exponent:int = None
        self.number_classes:int = 0
        self.case_activity_key: str=""
        self.case_timestamp_key:str =""
        self.case_id_key:str = ""
    def asdict(self):
        """
        used for exporting as a dictionary 
        """
        return {
            "time_precision": self.time_precision.name,
            "seq_len": self.seq_len,
            "emb_dim": self.emb_dim,
            "hid_dim": self.hid_dim,
            "mlp_dim": self.mlp_dim,
            "alpha": self.alpha,
            "dropout": self.dropout,
            "batch_size": self.batch_size,
            "lr": self.lr,
            "epochs": self.epochs,
            "importance_weight": self.importance_weight,
            "verbose_step": self.verbose_step,
            "cuda": self.cuda,
            "absolute_frequency_distribution": dict(self.absolute_frequency_distribution),
            "case_id_le": self.encoder_to_dict(self.case_id_le),
            "activity_le": self.encoder_to_dict(self.activity_le),
            "exponent": self.exponent,
            "number_classes": self.number_classes,
            "case_id_key": self.case_id_key,
            "case_timestamp_key":self.case_timestamp_key, 
            "case_activity_key": self.case_activity_key
        }
    def load_config(self, dic):
        """
        used for importing
        """
        self.time_precision= time_precision.TimePrecision[dic["time_precision"]] 
        self.seq_len=int(dic["seq_len"])
        self.emb_dim=int(dic["emb_dim"])
        self.hid_dim=int(dic["hid_dim"])
        self.mlp_dim=int(dic["mlp_dim"])
        self.alpha=float(dic["alpha"])
        self.dropout=float(dic["dropout"])
        self.batch_size=int(dic["batch_size"])
        self.lr=float(dic["lr"])
        self.epochs=int(dic["epochs"])
        self.importance_weight =dic["importance_weight"] #string
        self.verbose_step = int(dic["verbose_step"])
        self.cuda = True if dic["cuda"]=="True" else False  
        self.absolute_frequency_distribution = Counter(["absolute_frequency_distribution"])
        self.case_id_le = self.dict_to_encoder(dic["case_id_le"])
        self.activity_le = self.dict_to_encoder(dic["activity_le"])
        self.exponent =int(dic["exponent"])
        self.number_classes =int(dic["number_classes"])
        self.case_activity_key = dic["case_activity_key"]
        self.case_id_key = dic["case_id_key"]
        self.case_timestamp_key = dic["case_timestamp_key"]

    def encoder_to_dict(self, encoder:LabelEncoder)->dict:
        """
        cast the encoder to a dictionary 
        """
        return {label:index for index, label in enumerate(encoder.classes_)} 

    def dict_to_encoder(self, dic:dict)->LabelEncoder:
        """
        cast the dictionary to an encoder
        """

        encoder = LabelEncoder()
        encoder.classes_ = np.array(list(dic.keys()))
        return encoder

class NNManagement: 
    """
    This is the NNMamangement class. 
    
    Provided functinality: 
        - Train the model based on the event log. 
        - Test the model based on the event log.
        - Set params. 
        
    """
    def __init__(self, config:Config|None = None):
        self.config = Config() if config == None else config
        self.f1 = None
        self.recall= None
        self.acc = None
        self.time_error = None


    def evaluate(self):
        """
        This is the testing function for the model. It prints out the time_error, precision, recall, and f1 score.

        Returns:
            time_error (float): The time error.
            acc (float): The accuracy.
            recall (float): The recall.
            f1 (float): The F1 score.
        """
        #: main testing function
        self.model.eval()
        pred_times, pred_events = [], [] #inputs/training data
        gold_times, gold_events = [], [] #targets
        
        for i, batch in enumerate(tqdm(self.test_loader)):
            #batch: pair with two tensors, each containing respectively the time and event data.  
            gold_times.append(batch[0][:, -1].numpy()) # extract for each sequence the last time stamp/ the last event
            gold_events.append(batch[1][:, -1].numpy())
            pred_time, pred_event = self.model.predict(batch)
            if np.isnan(pred_times).any():
                raise exceptions.NaNException()
            pred_times.append(pred_time)
            pred_events.append(pred_event)

        pred_times = np.concatenate(pred_times).reshape(-1)
        print(type(pred_times))
        gold_times = np.concatenate(gold_times).reshape(-1)
        pred_events = np.concatenate(pred_events).reshape(-1)
        gold_events = np.concatenate(gold_events).reshape(-1)
        self.time_error = RMTPP_torch.abs_error(pred_times, gold_times)  #compute errors

        self.acc, self.recall, self.f1 = RMTPP_torch.clf_metric(pred_events, gold_events, n_class=self.config.number_classes) #get the metrics
        print(f"time_error: {self.time_error}, PRECISION: {self.acc}, RECALL: {self.recall}, F1: {self.f1}")
        
        return self.time_error, self.acc, self.recall, self.f1

    def get_training_statistics(self):
        """
        Returns:
            str: The accuracy, recall, and F1 score as a JSON object in string format.
        """
        if self.acc == None and self.recall == None and self.f1 ==None: 
            raise exceptions.ModelNotTrainedYet()

        #: dumps generates a string
        return {
            "time error": str(self.time_error),
            "acc":self.acc, 
            "recall":self.recall,
            "f1":self.f1
        }

    def import_nn_model(self, path):
        """
        imports a .pt file
        """
        saved_model_contents = torch.load(path)
        config = saved_model_contents['config']
        lossweight = saved_model_contents['lossweight']
        self.model = RMTPP_torch.Net(config, lossweight)
        if config.cuda: 
            self.model.to("cuda")
        else:
            self.model.to("cpu")
        self.model.load_state_dict(saved_model_contents['model_state_dict'])
        self.model.optimizer.load_state_dict(saved_model_contents['optimizer_state_dict'])
        self.model.eval() # relevant for droput layers.


    def export_nn_model(self, name="trained_model.pt"):
        """
        generates the .pt file containing the generated
        model. 

        model state dict contains
        optimizer state dict
        """
        dic = {
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.model.optimizer.state_dict(),
            'config': self.model.config, 
            'lossweight': self.model.lossweight
        }
        torch.save(dic, name)

    def random_search(self, search_parameters, iterations): 
        """
        Random search for the best hyperparameters. Saves the best model in the class.

        We only do this for the hid_dim, mlp_dim and emb_dim parameters. (decided arbitrarily, can be extended to other parameters as well.)

        Args:
            search_parameters (dict): Dictionary containing the search parameters.
                - 'hid_dim': [start, end]
                - 'mlp_dim': [start, end]
                - 'emb_dim': [start, end]
            iterations (int): Number of iterations.

        Returns:
            float: The best accuracy.
        """
        acc = 0
        best_model = None
        # self.load_data(train, test, case_id_key, timestamp_key, case_activity_key)
        for i in range(iterations): 
            a=random.randint(search_parameters["hid_dim"][0], search_parameters["hid_dim"][1])
            b=random.randint(search_parameters["mlp_dim"][0], search_parameters["mlp_dim"][1])
            c=  random.randint(search_parameters["emb_dim"][0], search_parameters["emb_dim"][1])
            self.config.hid_dim = a
            self.config.emb_dim= b
            self.config.mlp_dim=c
            self.train()
            if self.acc> acc: 
                self.config.hid_dim = a
                self.config.emb_dim= b
                self.config.mlp_dim=c
                acc = self.acc
                best_model = self.model
        self.model = best_model
        print(f"best accuracy: {acc}")
        return acc

    def grid_search(self, search_parameters): 
        """
        Grid search for the best hyperparameters.

        We only do this for the hid_dim, mlp_dim and emb_dim parameters. (decided arbitrarily, can be extended to other parameters as well.)

        Args:
            search_parameters (dict): Dictionary containing the search parameters.
            - 'hid_dim': [start, end, step]
            - 'mlp_dim': [start, end, step]
            - 'emb_dim': [start, end, step]

        Returns:
            float: The best accuracy.
        """
        acc = 0
        best_model = None
        # self.load_data(train, test, case_id_key, timestamp_key, case_activity_key)
        for i in range(search_parameters["hid_dim"][0], search_parameters["hid_dim"][1], search_parameters["hid_dim"][2]): 
                self.config.hid_dim =i 
                for j in range(search_parameters["mlp_dim"][0], search_parameters["mlp_dim"][1], search_parameters["mlp_dim"][2]): 
                    self.config.mlp_dim=j
                    for k in range(search_parameters["emb_dim"][0], search_parameters["emb_dim"][1], search_parameters["emb_dim"][2]):
                        self.config.emb_dim=k
                        self.config.our_implementation = True
                        self.train()
                        best_model = self.model
                        if self.acc> acc: 
                            acc = self.acc
        self.model = best_model
        print(f"best acc: {acc}")
        return acc
    
    def load_data(self,train_data, test_data, case_id, timestamp_key, event_key ):
        """
        imports a training and testing sublogs, which were preprocessed by the preprocessing module.

        it applies the sliding window algorithm to have subsequences of the same fixed length. 
        The output is passed to the respective DataLoader object, which computes the time differences 
        and casts the input to tensors;  generates the batches.
        """

        # apply sliding window
        train_set = RMTPP_torch.ATMDataset(self.config ,train_data, case_id,   timestamp_key, event_key ) 
        test_set =  RMTPP_torch.ATMDataset(self.config , test_data, case_id, timestamp_key, event_key)
        # generate time differences and load the data to torch tensors and generate the batches. 
        self.train_loader = DataLoader(train_set, batch_size=self.config.batch_size, shuffle=True, collate_fn= RMTPP_torch.ATMDataset.to_features)
        self.test_loader = DataLoader(test_set, batch_size=self.config.batch_size, shuffle=False, collate_fn= RMTPP_torch.ATMDataset.to_features)

        #: initialize a matrix to store the importance weights that will be passed to the CrossEntropyLoss object. 
        self.weight = np.ones(self.config.number_classes)
        if self.config.importance_weight:
            self.weight = train_set.importance_weight(self.config.absolute_frequency_distribution)
        


    def train(self):
            """
            This is the main training function.

            Args:
                train_data (DataFrame): The training data.
                test_data (DataFrame): The test data.
                case_id (str): The column name of the case ID in the data.
                timestamp_key (str): The key of the timestamp in the data.
                no_classes (int): The number of known markers.
            """
            # we already pass the split data to de ATM loader. ATMDAtaset uses the sliding window for generating the input for training.
            # since we are using tensors for training the sequence length remains fixed in each epoch, hence we cannot do "arbitrary length cuts" 
            # to the training data
            self.model =  RMTPP_torch.Net(self.config, lossweight=self.weight) #crete a NN instance
            self.model.set_optimizer(total_step=len(self.train_loader) * self.config.epochs) 


            if self.config.cuda: 
                self.model.cuda() #GPU 

            for epc in range(self.config.epochs): #run the epochs
                self.model.train()  
                range_loss1 = range_loss2 = range_loss = 0
                for i, batch in enumerate(tqdm(self.train_loader)):
                    
                    l1, l2, l = self.model.train_batch(batch) 
                    range_loss1 += l1
                    range_loss2 += l2
                    range_loss += l

                    if (i + 1) % self.config.verbose_step == 0:
                        print("time loss: ", range_loss1 / self.config.verbose_step)
                        print("event loss:", range_loss2 / self.config.verbose_step)
                        print("total loss:", range_loss / self.config.verbose_step)
                        range_loss1 = range_loss2 = range_loss = 0


            self.evaluate()
