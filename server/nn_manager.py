
import pm4py
import os
import pandas as pd
import pydoc_markdown
from ERPP_RMTPP import * 
from torch.utils.data import DataLoader
from tqdm import tqdm
import numpy as np




class NNManagement: 
    """
    This is the NNMamangement class. Provided functinality: 
    - ... tbd
    """
    def __init__(self):
        """
        --model defaults to RMTPP
        params: 
        :param seq_length: --seq_len determines the "b" constant
        that was defined in the paper (see 5.2 parameter learning), 
        determines a window size to save the training sequences into 
        in a tensor. 
        :param emb_dim: embedding dimension (--emb_dim)
        :param hid_dim: --hid_dim dimension for the hidden 
        dimension 
        :param mlp_dim: --mlp_dim dimension for the mlp (LSTM)
        :param alpha: --alpha=0.05 
        :param dropout: dropout parameter (RNN)
        :param batch_size: batch size
        :param lr: learning rate
        :param epochs: no of epochs
        """
        self.seq_length=10
        self.emb_dim=10
        self.hid_dim=32
        self.mlp_dim= 16
        self.alpha= 0.05
        self.dropout= 0.1
        self.batch_size= 1024
        self.lr= 1e-3
        self.epochs= 30
        self.model = None

    
    def set_training_parameters(self):
        pass


    def evaluate(self):
        self.model.eval()
        pred_times, pred_events = [], []
        gold_times, gold_events = [], []
        """
        pred son las predicciones
        gold son los targets 
        """


        for i, batch in enumerate(tqdm(test_loader)):
            """
            batch: pair with two tensors, each containing respectively
            the time and event data.  

            """
            """
            extract for each sequence the last time stamp/ the last
            event
            """
            gold_times.append(batch[0][:, -1].numpy())
            gold_events.append(batch[1][:, -1].numpy())
            pred_time, pred_event = model.predict(batch)
            time.sleep(7)



            pred_times.append(pred_time)
            pred_events.append(pred_event)
        pred_times = np.concatenate(pred_times).reshape(-1)
        gold_times = np.concatenate(gold_times).reshape(-1)
        pred_events = np.concatenate(pred_events).reshape(-1)
        gold_events = np.concatenate(gold_events).reshape(-1)
        time_error = abs_error(pred_times, gold_times) 
        """
        simplemente la diferencia de error 
        TODO: sospecho que esto lo hace con substraccion de matrices --> 
        ver funcionde numpy 
        """

        """
        con esto calcula las metricas    
        """
        acc, recall, f1 = clf_metric(pred_events, gold_events, n_class=config.event_class)
        print(f"epoch {epc}")
        print(f"time_error: {time_error}, PRECISION: {acc}, RECALL: {recall}, F1: {f1}")


    def train(self):
        config = {
            "seq_length": self.seq_length ,
            "emb_dim":self.emb_dim,
            "hid_dim": self.hid_dim,
            "mlp_dim":self.mlp_dim ,
            "alpha": self.alpha,
            "dropout": self.dropout,
            "batch_size": self.batch_size,
            "lr":self.lr ,
            "epochs": self.epochs
        }
    
        """
        con estas dos funciones importa los CSV: (aqui ya estan separados en train y test)
        """
        train_set = ATMDataset(config, subset='train')
        test_set = ATMDataset(config, subset='test')

        """
        dataloader es basicament ela funcion para cargar los datos!

        """
        train_loader = DataLoader(train_set, batch_size=config.batch_size, shuffle=True, collate_fn=ATMDataset.to_features)
        test_loader = DataLoader(test_set, batch_size=config.batch_size, shuffle=False, collate_fn=ATMDataset.to_features)

        weight = np.ones(config.event_class)
        if config.importance_weight:
            weight = train_set.importance_weight()
            print("importance weight: ", weight)
        """
        con esto crea una instancia de la net 
        """
        self.model = Net(config, lossweight=weight)

        model.set_optimizer(total_step=len(train_loader) * config.epochs, use_bert=True)
        model.cuda() #GPU TODO: rev docu

        for epc in range(config.epochs):
            model.train() #heredado de nn.Module 
            range_loss1 = range_loss2 = range_loss = 0
            for i, batch in enumerate(tqdm(train_loader)):
                l1, l2, l = model.train_batch(batch) #funcion definida para entrenar
                range_loss1 += l1
                range_loss2 += l2
                range_loss += l

                if (i + 1) % config.verbose_step == 0:
                    print("time loss: ", range_loss1 / config.verbose_step)
                    print("event loss:", range_loss2 / config.verbose_step)
                    print("total loss:", range_loss / config.verbose_step)
                    range_loss1 = range_loss2 = range_loss = 0
        self.evaluate()