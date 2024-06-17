"""
this neural network model is based on the paper
"Recurrent Marked Temporal Point Processes: Embedding Event History to Vector" by 
Du, et al. 
In particular, the implementation is a modified version from the repository
`https://github.com/woshiyyya/ERPP-RMTPP.git`.
"""
import torch
from torch import nn
from torch.optim import Adam
import numpy as np

class Net(nn.Module):
    def __init__(self, config, lossweight ):
        super(Net, self).__init__()
        self.config = config 
        self.n_class = config.number_classes
        self.embedding = nn.Embedding(num_embeddings=config.number_classes, embedding_dim=config.emb_dim)
        self.emb_drop = nn.Dropout(p=config.dropout)
        #droputs transform some random entries into zeros. Its used for regularization
        self.lstm = nn.LSTM(input_size=config.emb_dim + 1,         #we add one to the input because we merge the embedding vector with the time input (float)
                            hidden_size=config.hid_dim,
                            batch_first=True,
                            bidirectional=False) #TODO: check batch_first 
        self.mlp =   nn.Linear(in_features=config.hid_dim, out_features=config.mlp_dim) #TODO: check if this is the LSTM gate connection
        self.mlp_drop = nn.Dropout(p=config.dropout)

        self.event_linear = nn.Linear(in_features=config.mlp_dim, out_features=config.number_classes) #here we generate the output logits for the label
        self.time_linear = nn.Linear(in_features=config.mlp_dim, out_features=1) #here we calc the time prediction
        self.set_criterion(lossweight) 
        self.lossweight = lossweight
        self.optimizer = Adam(self.parameters(), lr=self.config.lr) #the author uses BertAdam, but we will just use Adam 

    def set_optimizer(self, total_step):
        self.optimizer = Adam(self.parameters(), lr=self.config.lr)


    def set_criterion(self, weight):
        #: cross entropy for the markers/label logits
        self.event_criterion = nn.CrossEntropyLoss(weight=torch.FloatTensor(weight))
        
        if self.config.cuda: 
            self.intensity_w = nn.Parameter(torch.tensor(0.1, dtype=torch.float, device='cuda'))
            self.intensity_b = nn.Parameter(torch.tensor(0.1, dtype=torch.float, device='cuda'))
            self.time_criterion = self.RMTPPLoss
        else: 
            self.intensity_w = nn.Parameter(torch.tensor(0.1, dtype=torch.float, device= 'cpu'))
            self.intensity_b = nn.Parameter(torch.tensor(0.1, dtype=torch.float, device = 'cpu'))
            self.time_criterion = self.RMTPPLoss
        

    def RMTPPLoss(self, pred, gold):
        """
        calculate the loss for the time. 
        """
        loss = torch.mean(pred + self.intensity_w * gold + self.intensity_b +
                          (torch.exp(pred + self.intensity_b) -
                           torch.exp(pred + self.intensity_w * gold + self.intensity_b)) / self.intensity_w) #TODO: this should be the loss function for the time prediction; that is why we have all these exp. (see paper)
        return -1 * loss



    def forward(self, input_time, input_events):
        event_embedding = self.embedding(input_events)
        event_embedding = self.emb_drop(event_embedding)

        # merge the embed vector with the time input (extra row)
        lstm_input = torch.cat((event_embedding, input_time.unsqueeze(-1)), dim=-1)
        hidden_state, _ = self.lstm(lstm_input) 

        # hidden_state = torch.cat((hidden_state, input_time.unsqueeze(-1)), dim=-1) THIS WAS COMMENTED FROM BEFORE
        mlp_output = torch.tanh(self.mlp(hidden_state[:, -1, :]))  #multi layer perceptorn output
        mlp_output = self.mlp_drop(mlp_output) 
        #Here we are basically passing the output through TWO DIFFERENT LAYERS separately.
        event_logits = self.event_linear(mlp_output)  # the output is separated and passed to a specific activation function
        time_logits = self.time_linear(mlp_output)  # the output is separated and passed to a specific activation function
        return time_logits, event_logits  # get the predictions 

    def dispatch(self, tensors):
        if self.config.cuda:
            for i in range(len(tensors)):
                
                tensors[i] = tensors[i].cuda().contiguous()
        else:
            for i in range(len(tensors)):
                tensors[i] = tensors[i].contiguous()
        return tensors


    def train_batch(self, batch):
        time_tensor, event_tensor = batch

        #here we make sure to REMOVE THE LABEL from the training input. that is why we do "slicing"
        time_input, time_target = self.dispatch([time_tensor[:, :-1], time_tensor[:, -1]])
        event_input, event_target = self.dispatch([event_tensor[:, :-1], event_tensor[:, -1]])
        time_logits, event_logits = self.forward(time_input, event_input)
        #calc loss
        loss1 = self.time_criterion(time_logits.view(-1), time_target.view(-1))
        loss2 = self.event_criterion(event_logits.view(-1, self.n_class), event_target.view(-1))
        loss = self.config.alpha * loss1 + loss2  #total loss formula 
        loss.backward() #backpropagation trhough time.

        self.optimizer.step()
        self.optimizer.zero_grad() #reset grads
        return loss1.item(), loss2.item(), loss.item()

    
    def predict(self, batch, pm_active = False):
        """
        make a prediction
        :param batch: a batch containing possible more than 
        :param pm_active: returns only the most likely prediction
        one input for doing one or more predictions
        :return:  returns two lists. the first one contains
        the timestamps of the predictions
        the second list the index (encoded marker) of the event that has the
        highest probability.
        """
        time_tensor, event_tensor = batch
        #make sure to cut out the last event/timestamp from each sequence: 
        time_input, time_target = self.dispatch([time_tensor[:, :-1], time_tensor[:, -1]])
        event_input, event_target = self.dispatch([event_tensor[:, :-1], event_tensor[:, -1]])
        time_logits, event_logits = self.forward(time_input, event_input)
        event_pred=  event_logits.detach().cpu().numpy()
        event_pred = np.argmax(event_pred, axis=-1) #for each label find the index that maximizes the pred.
        if pm_active: 
            #: in case we just need the most likely prediction
            lst = event_logits.detach().cpu().numpy().tolist()
            last = softmax(lst[-1]).tolist() #get event logits from last run
            max_prob = max(last)
            event_index = last.index(max_prob) #compute argmax 
            time_pred = time_logits.detach().cpu().numpy().tolist() #: get last time prdiction
            return event_index, max_prob,time_pred[-1][-1]

        
        time_pred = time_logits.detach().cpu().numpy()
        return time_pred, event_pred


    def predict_sorted(self, batch): 
        """
        make a prediction
        :param batch: a batch containing possible more than 
        one input for doing one or more predictions
        :return: returns two lists. the first one contains
        the timestamps of the predictions
        the second list contains tuples of form `(probability, event_index)`
        this second list is sorted in descending order. Hence the 
        first event has the highest probability. The `event_index` corresponds
        to the encoding of a marker.
        """
        time_tensor, event_tensor = batch
        #make sure to cut out the last event/timestamp from each sequence: 
        time_input, time_target = self.dispatch([time_tensor[:, :-1], time_tensor[:, -1]])
        event_input, event_target = self.dispatch([event_tensor[:, :-1], event_tensor[:, -1]])
        time_logits, event_logits = self.forward(time_input, event_input)
        
        event_pred = event_logits.detach().cpu().numpy().tolist()
        event_pred_with_indices= []
        for index,  logit_list in enumerate(event_pred): 
            index_list= []
            for event_index, prediction in enumerate(softmax(logit_list)):
                index_list.append((prediction, event_index))
            index_list.sort(reverse=True)
            event_pred_with_indices.append(index_list)
        time_pred = time_logits.detach().cpu().tolist()
        return time_pred, event_pred_with_indices



def softmax(x): 
    x = np.exp(x)
    return x / x.sum()