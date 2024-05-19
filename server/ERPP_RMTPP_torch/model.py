import torch
from torch import nn
from torch.optim import Adam, SGD
import numpy as np



class Net(nn.Module):
    def __init__(self, config, lossweight):
        super(Net, self).__init__()
        self.config = config 
        self.n_class = config.event_class
        self.embedding = nn.Embedding(num_embeddings=config.event_class, embedding_dim=config.emb_dim)
        self.emb_drop = nn.Dropout(p=config.dropout)
        #droputs transform some random entries into zeros. Its used for regularization
        self.lstm = nn.LSTM(input_size=config.emb_dim + 1,         #we add one to the input because we merge the embedding vector with the time input (float)
                            hidden_size=config.hid_dim,
                            batch_first=True,
                            bidirectional=False) #TODO: check batch_first 
        self.mlp =   nn.Linear(in_features=config.hid_dim, out_features=config.mlp_dim) #TODO: check if this is the LSTM gate connection
        self.mlp_drop = nn.Dropout(p=config.dropout)

        self.event_linear = nn.Linear(in_features=config.mlp_dim, out_features=config.event_class) #here we generate the output logits for the label
        self.time_linear = nn.Linear(in_features=config.mlp_dim, out_features=1) #here we calc the time prediction
        self.set_criterion(lossweight) 
        #self.optimizer = Adam(self.parameters(), lr=self.config.lr) #the author uses BertAdam, but we will just use Adam 
        self.optimizer = SGD(self.parameters(), lr=self.config.lr) #the author uses BertAdam, but we will just use Adam 
        #TODO: add choice of BertAdam/ SGD

    def set_optimizer(self, total_step, use_bert=False):
        self.optimizer = Adam(self.parameters(), lr=self.config.lr)


    def set_criterion(self, weight):
        self.event_criterion = nn.CrossEntropyLoss(weight=torch.FloatTensor(weight))
        #cross entropy for the markers/label logits

        if self.config.model == 'rmtpp':
            if self.config.cuda: 
                self.intensity_w = nn.Parameter(torch.tensor(0.1, dtype=torch.float, device='cuda'))
                self.intensity_b = nn.Parameter(torch.tensor(0.1, dtype=torch.float, device='cuda'))
                self.time_criterion = self.RMTPPLoss
            else: 
                self.intensity_w = nn.Parameter(torch.tensor(0.1, dtype=torch.float))
                self.intensity_b = nn.Parameter(torch.tensor(0.1, dtype=torch.float))
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
        #print("input time shape: ", input_time.shape) #debug
        #print(input_time)
        #print(input_events)
        event_embedding = self.embedding(input_events)
        #print("event embedding: ", event_embedding) #debug
        #print(event_embedding)
        event_embedding = self.emb_drop(event_embedding)

        #print("event embedding: ", event_embedding) #debug

        # merge the embed vector with the time input (extra row)
        lstm_input = torch.cat((event_embedding, input_time.unsqueeze(-1)), dim=-1)
        hidden_state, _ = self.lstm(lstm_input) 

        # hidden_state = torch.cat((hidden_state, input_time.unsqueeze(-1)), dim=-1) THIS WAS COMMENTED FROM BEFORE
        mlp_output = torch.tanh(self.mlp(hidden_state[:, -1, :])) 
        mlp_output = self.mlp_drop(mlp_output) 
        """
        Here we are basically passing the output through TWO DIFFERENT LAYERS separately.
        """
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

        # print("--"*20)
        #here we make sure to REMOVE THE LABEL from the training input. that is why we do "slicing"
        time_input, time_target = self.dispatch([time_tensor[:, :-1], time_tensor[:, -1]])
        event_input, event_target = self.dispatch([event_tensor[:, :-1], event_tensor[:, -1]])
        #print("time input: ", time_input)
        time_logits, event_logits = self.forward(time_input, event_input)
        #print("time logits: ", time_logits)
        # print(time_logits)
        # print(event_logits)
        # print("^^^"*20)
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
        event_pred=  event_logits.detach().cpu().numpy()
        event_pred = np.argmax(event_pred, axis=-1) #for each label find the index that maximizes the pred.
        time_pred = time_logits.detach().cpu().numpy()
        #print(event_pred.sort(axis= -1))
        return time_pred, event_pred


    def predict_get_sorted(self, batch): 
        time_tensor, event_tensor = batch
        #make sure to cut out the last event/timestamp from each sequence: 
        time_input, time_target = self.dispatch([time_tensor[:, :-1], time_tensor[:, -1]])
        event_input, event_target = self.dispatch([event_tensor[:, :-1], event_tensor[:, -1]])
        time_logits, event_logits = self.forward(time_input, event_input)
        
        event_pred = event_logits.detach().cpu().numpy().tolist()
        event_pred_with_indices= []
        for index,  prediction_list in enumerate(event_pred): 
            index_list= []
            for event_index, prediction in enumerate(prediction_list):
                index_list.append((prediction, event_index))
                index_list.sort()
            event_pred_with_indices.append(index_list)
            index_list.sort()
        time_pred = time_logits.detach().cpu().tolist()
        return time_pred, event_pred_with_indices

