"""
This module is in charge of training the NN Model
"""
import pm4py
import os
import time
import pandas as pd
from server import RMTPP_torch
from server import loggers
from server import exceptions
from torch.utils.data import DataLoader
from tqdm import tqdm
import numpy as np
import json
import logging
import torch
from collections import Counter
import random
import pprint
from sklearn.preprocessing import LabelEncoder

class Config: 
    def __init__(self):
        self.seq_len= 10
        self.emb_dim= 32
        self.hid_dim=32
        self.mlp_dim= 16
        self.alpha= 0.05
        self.dropout= 0.1
        self.batch_size= 1024
        self.lr= 1e-3
        self.epochs= 1 
        self.importance_weight = "store_true"
        self.verbose_step = 350
        self.cuda = False
        self.absolute_frequency_distribution = Counter()
        self.case_id_le = None
        self.activity_le = None
        self.exponent = None
        self.number_classes = 0
        self.train_time_limit = None
        self.case_activity_key=""
        self.case_timestamp_key=""
        self.case_id_key = ""
    def asdict(self):
        return {
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

    def encoder_to_dict(self, encoder):
        return {label:index for index, label in enumerate(encoder.classes_)} 

    def dict_to_encoder(self, dic):
        encoder = LabelEncoder()
        encoder.classes_ = np.array(list(dic.keys()))
        return encoder

class NNManagement: 
    """
    This is the NNMamangement class. Provided functinality: 
    - train the model based on the event log. 
    - test the model based on the event log.
    - set params. 
    - TODO: might be extended 
    """
    def __init__(self, config = None):
        self.config = Config() if config == None else config
        self.f1 = None
        self.recall= None
        self.acc = None
        self.time_error = None



    def set_training_parameters(self,  params):
        """
        Used for setting the training parameters. Note that not all params must be input.
        :param params: dictionary containing the parameters, the following keys are possible:

        * seq_length: --seq_len determines the "b" constant that was defined in the paper (see 5.2 parameter learning)
        determines a window size to save the training sequences into in a tensor. 
        * emb_dim: embedding dimension 
        * hid_dim: --hid_dim dimension for the hidden dimension 
        * mlp_dim: --mlp_dim dimension for the mlp (LSTM) TODO: revise
        * alpha: --alpha=0.05 
        * dropout: dropout parameter (RNN)
        * batch_size: batch size
        * lr: learning rate
        * epochs: no of epochs
        * importance_weight: importance weight for the loss function
        * verbose_step: after how many steps the loss should be printed
        * train_time_limit: time limit for training in minutes, when the time is over training will be aborted
        """
        self.config.seq_len = params.get('seq_len')
        self.config.emb_dim = params.get('emb_dim')
        self.config.hid_dim = params.get('hid_dim')
        self.config.mlp_dim = params.get('mlp_dim')
        self.config.alpha = params.get('alpha')
        self.config.dropout = params.get('dropout')
        self.config.batch_size = params.get('batch_size')
        self.config.lr = params.get('lr')
        self.config.epochs = params.get('epochs')
        self.config.importance_weight = params.get('importance_weight')
        self.config.verbose_step = params.get('verbose_step')
        self.config.train_time_limit = params.get('train_time_limit')

    def evaluate(self):
        """
        this is the testing function for the model. 
        it prints out the time_error, precision, recall and f1 score.
        :return: time_error, acc, recall, f1
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
            pred_times.append(pred_time)
            pred_events.append(pred_event)

           

        pred_times = np.concatenate(pred_times).reshape(-1)
        gold_times = np.concatenate(gold_times).reshape(-1)
        pred_events = np.concatenate(pred_events).reshape(-1)
        gold_events = np.concatenate(gold_events).reshape(-1)
        self.time_error = RMTPP_torch.abs_error(pred_times, gold_times)  #compute errors


        self.acc, self.recall, self.f1 = RMTPP_torch.clf_metric(pred_events, gold_events, n_class=self.config.number_classes) #get the metrics
        
        print(f"time_error: {self.time_error}, PRECISION: {self.acc}, RECALL: {self.recall}, F1: {self.f1}")
        
        return self.time_error, self.acc, self.recall, self.f1

    def get_training_statistics(self):
        """
        :return: the accuracy, recall and f1 score 
        as a json object in string format. 
        """
        if self.acc == None and self.recall == None and self.f1 ==None: 
            raise exceptions.ModelNotTrainedYet()

        #: dumps generates a string
        return {
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

    def random_search(self,train,test,  search_parameters, iterations, case_id_key, timestamp_key, case_activity_key ): 
        acc = 0
        best_model = None
        self.load_data(train, test, case_id_key, timestamp_key, case_activity_key)
        for i in range(iterations): 
            a=random.randint(search_parameters["hidden_dim"][0], search_parameters["hidden_dim"][1])
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

    def grid_search(self,train,test,  search_parameters, case_id_key, timestamp_key, case_activity_key ): 
        acc = 0
        best_model = None
        self.load_data(train, test, case_id_key, timestamp_key, case_activity_key)
        for i in range(search_parameters["hidden_dim"][0], search_parameters["hidden_dim"][1], search_parameters["hidden_dim"][2]): 
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
        train_set = RMTPP_torch.ATMDataset(self.config ,train_data, case_id,   timestamp_key, event_key ) 
        test_set =  RMTPP_torch.ATMDataset(self.config , test_data, case_id, timestamp_key, event_key)
        # now load the data to torch tensors and generate the batches. also 
        self.train_loader = DataLoader(train_set, batch_size=self.config.batch_size, shuffle=True, collate_fn= RMTPP_torch.ATMDataset.to_features)
        self.test_loader = DataLoader(test_set, batch_size=self.config.batch_size, shuffle=False, collate_fn= RMTPP_torch.ATMDataset.to_features)

        #: initialize a matrix to store the importance weights
        # that will be passed to the CrossEntropyLoss object. 
        self.weight = np.ones(self.config.number_classes)
        if self.config.importance_weight:
            self.weight = train_set.importance_weight(self.config.absolute_frequency_distribution)
        


    def train(self):
        """
        This is the main training function 
        :param train_data: train data df
        :param test_data: test data df  
        :param case_id: case id column name in the df
        :param timestampt_key: timestamp key in the df
        :param no_classes: number of known markers.
        """
        # we already pass the split data to de ATM loader. ATMDAtaset uses the sliding window for generating the input for training.
        # since we are using tensors for training the sequence length remains fixed in each epoch, hence we cannot do "arbitrary length cuts" 
        # to the training data
        self.model =  RMTPP_torch.Net(self.config, lossweight=self.weight) #crete a NN instance
        self.model.set_optimizer(total_step=len(self.train_loader) * self.config.epochs) #TODO: fix use bert (doesnt exist)


        if self.config.cuda: 
            self.model.cuda() #GPU 

        start_time = time.time()
        for epc in range(self.config.epochs): #do the epochs
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

                # check if the training time limit is exceeded
                elapsed_time = time.time() - start_time
                if self.config.train_time_limit is not None and elapsed_time > self.config.train_time_limit * 60:
                    raise exceptions.TrainTimeLimitExceeded()

        self.evaluate()