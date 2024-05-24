
import pm4py
import os
import pandas as pd
import pydoc_markdown
from ERPP_RMTPP_torch import * 
from torch.utils.data import DataLoader
from tqdm import tqdm
import numpy as np
import json
from exceptions import *
import logging
from collections import Counter
from loggers import logger_evaluate

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
        self.model =None 
        self.importance_weight = "store_true"
        self.verbose_step = 350
        self.event_class = 0
        self.cuda = False
        self.absolute_frequency_distribution = Counter()

class NNManagement: 
    """
    This is the NNMamangement class. Provided functinality: 
    - train the model based on the event log. 
    - test the model based on the event log.
    - set params. 
    - TODO: might be extended 
    """
    def __init__(self):
        self.config = Config()
        self.f1 = None
        self.recall= None
        self.acc = None
        self.absolute_frequency_distribution =None
        self.time_error = None



    def set_training_parameters(self,  params):
        """
        Used for setting the training parameters. Note that not all params must be input.

        :param seq_length: --seq_len determines the "b" constant that was defined in the paper (see 5.2 parameter learning)
        determines a window size to save the training sequences into in a tensor. 
        :param emb_dim: embedding dimension 
        :param hid_dim: --hid_dim dimension for the hidden dimension 
        :param mlp_dim: --mlp_dim dimension for the mlp (LSTM) TODO: revise
        :param alpha: --alpha=0.05 
        :param dropout: dropout parameter (RNN)
        :param batch_size: batch size
        :param lr: learning rate
        :param epochs: no of epochs
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

    def evaluate(self, config):
        """
        this is the testing function for the model. 
        it prints out the time_error, precision, recall and f1 score.
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
        self.time_error = abs_error(pred_times, gold_times)  #compute errors


        self.acc, self.recall, self.f1 = clf_metric(pred_events, gold_events, n_class=config.event_class) #get the metrics
        
        print(f"time_error: {self.time_error}, PRECISION: {self.acc}, RECALL: {self.recall}, F1: {self.f1}")

    def get_training_statistics(self):
        """
        :return: the accuracy, recall and f1 score 
        as a json object in string format. 
        """
        if self.acc == None and self.recall == None and self.f1 ==None: 
            raise ModelNotTrainedYet()

        #: dumps generates a string
        return json.dumps({
            "acc":self.acc, 
            "recall":self.recall,
            "f1":self.f1
        })

    def import_nn_model(self, path):
        """
        imports a .pt file
        """

        weight = np.ones(self.config.event_class)
        if self.config.importance_weight:
            weight = train_set.importance_weight(self.config.absolute_frequency_distribution)
        self.model = Net(self.config, lossweight={}) #crete a NN instance
        self.model.load_state_dict(torch.load(path))
        print(type(self.model))
        self.model.eval() # relevant for droput layers.


    def export_nn_model(self):
        """
        generates the .pt file containing the generated
        model. 
        """
        torch.save(self.model.state_dict(), "model.pt") 



    def train(self, train_data, test_data, case_id, timestamp_key, event_key, no_classes):
        """
        This is the main training function 
        :param train_data: train data df
        :param test_data: test data df  
        :param case_id: case id column name in the df
        :param timestampt_key: timestamp key in the df
        :param no_classes: number of known markers.
        """

        self.config.event_class = no_classes

        # we already pass the split data to de ATM loader. ATMDAtaset uses the sliding window for generating the input for training.
        # since we are using tensors for training the sequence length remains fixed in each epoch, hence we cannot do "arbitrary length cuts" 
        # to the training data

        train_set = ATMDataset(self.config ,train_data, case_id,   timestamp_key, event_key ) 
        test_set = ATMDataset(self.config , test_data, case_id, timestamp_key, event_key)

        time_seqs_values = [set(l) for l in test_set.time_seqs]

        # now load the data to torch tensors and generate the batches. also 
       
        self.train_loader = DataLoader(train_set, batch_size=self.config.batch_size, shuffle=True, collate_fn=ATMDataset.to_features)
        
        self.test_loader = DataLoader(test_set, batch_size=self.config.batch_size, shuffle=False, collate_fn=ATMDataset.to_features)

        #: initialize a matrix to store the importance weights
        # that will be passed to the CrossEntropyLoss object. 
        weight = np.ones(self.config.event_class)
        if self.config.importance_weight:
            weight = train_set.importance_weight(self.config.absolute_frequency_distribution)
        
        self.model = Net(self.config, lossweight=weight) #crete a NN instance
        self.model.set_optimizer(total_step=len(self.train_loader) * self.config.epochs) #TODO: fix use bert (doesnt exist)

        if self.config.cuda: 
            self.model.cuda() #GPU 


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
        


        self.evaluate( self.config)