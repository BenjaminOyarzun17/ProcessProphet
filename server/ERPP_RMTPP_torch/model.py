import torch
from torch import nn
from torch.optim import Adam
import numpy as np
#from optimization import BertAdam



class Net(nn.Module):
    def __init__(self, config, lossweight):
        super(Net, self).__init__()
        self.config = config 
        self.n_class = config.event_class 
        self.embedding = nn.Embedding(num_embeddings=config.event_class, embedding_dim=config.emb_dim)
        self.emb_drop = nn.Dropout(p=config.dropout)
        #droputs transform some random entries into zeros. Its used for regularization
        self.lstm = nn.LSTM(input_size=config.emb_dim + 1,
                            hidden_size=config.hid_dim,
                            batch_first=True,
                            bidirectional=False) #TODO: check batch_first 
        #we add one to the input because we merge the embedding vector with the time input (float)
        self.mlp = nn.Linear(in_features=config.hid_dim, out_features=config.mlp_dim) #TODO: check if this is the LSTM gate connection
        self.mlp_drop = nn.Dropout(p=config.dropout)

        self.event_linear = nn.Linear(in_features=config.mlp_dim, out_features=config.event_class) #here we generate the output logits for the label

        self.time_linear = nn.Linear(in_features=config.mlp_dim, out_features=1) #here we calc the time prediction
        self.set_criterion(lossweight) 
        self.optimizer = Adam(self.parameters(), lr=self.config.lr) #the author uses BertAdam, but we will just use Adam 
        #TODO: add choice of BertAdam/ SGD

    def set_optimizer(self, total_step, use_bert=False):
        if use_bert:
            pass
            """
            self.optimizer = BertAdam(params=self.parameters(),
                                      lr=self.config.lr,
                                      warmup=0.1,
                                      t_total=total_step)
            """
        else:
            self.optimizer = Adam(self.parameters(), lr=self.config.lr)


    def set_criterion(self, weight):
        self.event_criterion = nn.CrossEntropyLoss(weight=torch.FloatTensor(weight))
        #cross entropy for the markers/label logits

        if self.config.model == 'rmtpp':
            self.intensity_w = nn.Parameter(torch.tensor(0.1, dtype=torch.float, device='cuda'))
            self.intensity_b = nn.Parameter(torch.tensor(0.1, dtype=torch.float, device='cuda'))
            self.time_criterion = self.RMTPPLoss
            # TODO: revise this params
        else:
            self.time_criterion = nn.MSELoss()

    def RMTPPLoss(self, pred, gold):
        loss = torch.mean(pred + self.intensity_w * gold + self.intensity_b +
                          (torch.exp(pred + self.intensity_b) -
                           torch.exp(pred + self.intensity_w * gold + self.intensity_b)) / self.intensity_w) #TODO: this should be the loss function for the time prediction; that is why we have all these exp. (see paper)
        return -1 * loss



    def forward(self, input_time, input_events):
        event_embedding = self.embedding(input_events)
        event_embedding = self.emb_drop(event_embedding)
        #merge the embed vector with the time input (extra row)
        lstm_input = torch.cat((event_embedding, input_time.unsqueeze(-1)), dim=-1)
        hidden_state, _ = self.lstm(lstm_input) 

        # hidden_state = torch.cat((hidden_state, input_time.unsqueeze(-1)), dim=-1) THIS WAS COMMENTED FROM BEFORE
        mlp_output = torch.tanh(self.mlp(hidden_state[:, -1, :])) 
        mlp_output = self.mlp_drop(mlp_output) 
        """
        aqui esta basicamente pasando el output por DOS LAYERS DIFERNETS; por separado. 
        """
        event_logits = self.event_linear(mlp_output)  #the ouptut is separated  and passed toa specific act func
        time_logits = self.time_linear(mlp_output) #the output is separaed and passed to a specific act func
        return time_logits, event_logits  #get the predictions 

    def dispatch(self, tensors):
        for i in range(len(tensors)):
            tensors[i] = tensors[i].cuda().contiguous()
        return tensors

    def train_batch(self, batch):
        time_tensor, event_tensor = batch

        print("--"*20)
        #here we make sure to REMOVE THE LABEL from the training input. that is why we do "slicing"
        time_input, time_target = self.dispatch([time_tensor[:, :-1], time_tensor[:, -1]])
        event_input, event_target = self.dispatch([event_tensor[:, :-1], event_tensor[:, -1]])
        time_logits, event_logits = self.forward(time_input, event_input)
        print(time_logits)
        print(event_logits)
        print("^^^"*20)
        #calc loss
        loss1 = self.time_criterion(time_logits.view(-1), time_target.view(-1))
        loss2 = self.event_criterion(event_logits.view(-1, self.n_class), event_target.view(-1))
        loss = self.config.alpha * loss1 + loss2  #total loss formula 
        loss.backward() #backpropagation trhough time.

        self.optimizer.step()
        self.optimizer.zero_grad() #reset grads
        return loss1.item(), loss2.item(), loss.item()

    
    def predict(self, batch):
        time_tensor, event_tensor = batch
        #make sure to cut out the last event/timestamp from each sequence: 
        time_input, time_target = self.dispatch([time_tensor[:, :-1], time_tensor[:, -1]])
        event_input, event_target = self.dispatch([event_tensor[:, :-1], event_tensor[:, -1]])
        time_logits, event_logits = self.forward(time_input, event_input)
        
        event_pred = np.argmax(event_logits.detach().cpu().numpy(), axis=-1) #pick the one label with max value
        time_pred = time_logits.detach().cpu().numpy()
        return time_pred, event_pred
