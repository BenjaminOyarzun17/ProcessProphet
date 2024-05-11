import torch
from torch import nn
from torch.optim import Adam
import numpy as np
#from optimization import BertAdam

class Net(nn.Module):
    """
    config son configuraciones que manualmente se pasan desde la terminal


    --name=EXPERIMENT_NAME 
               --model=    # "erpp" or "rmtpp"
               --seq_len=10 
               --emb_dim=10 
               --hid_dim=32 
               --mlp_dim=16 
               --alpha=0.05   # weight on time loss
               --dropout=0.1 
               --batch_size= 1024 
               --lr=1e-3 
               --epochs=30 
               --importance_weight # if use importance loss weight
               --verbose_step 
    """
    def __init__(self, config, lossweight):
        super(Net, self).__init__()
        self.config = config #ver comentario de arriba
        self.n_class = config.event_class 
        self.embedding = nn.Embedding(num_embeddings=config.event_class, embedding_dim=config.emb_dim)
        self.emb_drop = nn.Dropout(p=config.dropout)
        """
        esto (dropout) es mas que nada una buena practica de regularizacion; hace que ciertos entries
        se vuelvan cero (aleatoriamente). no prestar tanta atencion supongo.
        """
        self.lstm = nn.LSTM(input_size=config.emb_dim + 1,
                            hidden_size=config.hid_dim,
                            batch_first=True,
                            bidirectional=False) # reisar def de batch first
        # notar que se suma uno al input dimension pq le estamos agregando el tiempo!
        self.mlp = nn.Linear(in_features=config.hid_dim, out_features=config.mlp_dim)
        # TODO: revisar que es mlp SOSPECHO que es la conexion "recurrent". o sino 
        # refiere al "gate" de la LSTM --> revisar arquitectura de LSTM
        self.mlp_drop = nn.Dropout(p=config.dropout)



        self.event_linear = nn.Linear(in_features=config.mlp_dim, out_features=config.event_class)
        # con esto configura el num de outputs para las clases; en particular la activation function
        self.time_linear = nn.Linear(in_features=config.mlp_dim, out_features=1)
        # con esto configura el num de outputs para el tiempo y act function
        self.set_criterion(lossweight) #lossweight es parametro

    def set_optimizer(self, total_step, use_bert=True):
        if use_bert:
            self.optimizer = BertAdam(params=self.parameters(),
                                      lr=self.config.lr,
                                      warmup=0.1,
                                      t_total=total_step)
        else:
            self.optimizer = Adam(self.parameters(), lr=self.config.lr)
        """
        notar que este tipo usa Adam en vez de SGD. (en realidad por temas
        de tecnicos no deberia importar tanto esto ultimo)
        """


    def set_criterion(self, weight):
        self.event_criterion = nn.CrossEntropyLoss(weight=torch.FloatTensor(weight))
        """
        cross entropy para los clasificadores de los markers  
        """
        if self.config.model == 'rmtpp':
            self.intensity_w = nn.Parameter(torch.tensor(0.1, dtype=torch.float, device='cuda'))
            self.intensity_b = nn.Parameter(torch.tensor(0.1, dtype=torch.float, device='cuda'))
            self.time_criterion = self.RMTPPLoss
            # TODO: revisar que son estos parametros intensity w e intensity b
            # TODO: revisar time criterion


        else:
            self.time_criterion = nn.MSELoss()

    def RMTPPLoss(self, pred, gold):
        loss = torch.mean(pred + self.intensity_w * gold + self.intensity_b +
                          (torch.exp(pred + self.intensity_b) -
                           torch.exp(pred + self.intensity_w * gold + self.intensity_b)) / self.intensity_w)
        return -1 * loss
        """
        sospecho que con esto calcula la esperanza que se calcula en la ultima integral
        TODO: revisar si es el caso. 
        """



    def forward(self, input_time, input_events):
        event_embedding = self.embedding(input_events)
        event_embedding = self.emb_drop(event_embedding)
        """
        aqui basicamente esta juntando el input temporal y el embeded vector en un mismo tensor.
        me imagino que el merge lo hace a lo largo de la ultima dimension
        """
        lstm_input = torch.cat((event_embedding, input_time.unsqueeze(-1)), dim=-1)
        hidden_state, _ = self.lstm(lstm_input) #ahora le pasamos el input

        # hidden_state = torch.cat((hidden_state, input_time.unsqueeze(-1)), dim=-1) esto estaba comentado de antes
        mlp_output = torch.tanh(self.mlp(hidden_state[:, -1, :])) #define las transformaciones del output 
        mlp_output = self.mlp_drop(mlp_output) #hace un drop TODO: revisa que es un drop
        # sospecho que es la activation de la recurrent connection
        """
        aqui esta basicamente pasando el output por DOS LAYERS DIFERNETS; por separado. 
        """
        event_logits = self.event_linear(mlp_output) 
        time_logits = self.time_linear(mlp_output)
        return time_logits, event_logits   #con esto tenemos efectivamente dos outputs!

    def dispatch(self, tensors):
        for i in range(len(tensors)):
            tensors[i] = tensors[i].cuda().contiguous()
        return tensors

    def train_batch(self, batch):
        time_tensor, event_tensor = batch
        time_input, time_target = self.dispatch([time_tensor[:, :-1], time_tensor[:, -1]])
        event_input, event_target = self.dispatch([event_tensor[:, :-1], event_tensor[:, -1]])

        """
        justamente aqui hace la prediccion 
        """
        time_logits, event_logits = self.forward(time_input, event_input)
        """
        calcula las losses 
        """
        loss1 = self.time_criterion(time_logits.view(-1), time_target.view(-1))
        loss2 = self.event_criterion(event_logits.view(-1, self.n_class), event_target.view(-1))
        loss = self.config.alpha * loss1 + loss2  #con esto define la loss TOTAL
        loss.backward() #backpropagation basicamente.

        self.optimizer.step()
        self.optimizer.zero_grad() #resetea gradientes
        return loss1.item(), loss2.item(), loss.item()

    
    """
    con esto hacemos predicciones puntuales 
    
    """
    def predict(self, batch):
        time_tensor, event_tensor = batch
        """
        make sure to cut out the last event/timestamp from each sequence: 
        """
        time_input, time_target = self.dispatch([time_tensor[:, :-1], time_tensor[:, -1]])
        event_input, event_target = self.dispatch([event_tensor[:, :-1], event_tensor[:, -1]])
        time_logits, event_logits = self.forward(time_input, event_input)
        """
        escoge el que tenga max value 
        """
        event_pred = np.argmax(event_logits.detach().cpu().numpy(), axis=-1)
        time_pred = time_logits.detach().cpu().numpy()
        return time_pred, event_pred
