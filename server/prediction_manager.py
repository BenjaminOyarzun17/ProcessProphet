from loggers import logger_get_dummy_process,logger_single_prediction , logger_multiple_prediction
from exceptions import ProcessTooShort, SeqLengthTooHigh, NotOneCaseId
from preprocessing import Preprocessing
from RMTPP_torch import * 
from torch.utils.data import DataLoader
import pandas as pd
import numpy as np
import pprint 
import json
import time
import random



class PredictionManager: 
    def __init__(self, model, case_id_key, activity_key, timestamp_key, config ):
        """
        :param model: the model used for doing predictions
        :param case_id_key: case id key of the log 
        :param activity_key: activity key of the log
        :param timestamp_key: timestamp key of the log
        :param config: configuration used for training and important hyperparams.
        """
        self.model =model  #: we assume there is an already existing model
        self.case_id_key = case_id_key # we assume the three important columns are known
        self.activity_key = activity_key
        self.timestamp_key = timestamp_key
        self.config = config
        
        self.current_case_id = None
        self.paths = []
        self.decoded_paths= []
        self.encoded_df = None
        self.recursive_event_seqs = []
        self.recursive_time_seqs = []
        self.recursive_time_diffs= []
        self.end_activities= {}





    def get_dummy_process(self, df, case_id_column):
        """
        just used for testing; create a dummy input df.
        """
        case_ids = df[case_id_column].unique()
        selected_id = -1
        length = 0
        for id in case_ids: 
            length = len(df[df[case_id_column]==id])
            if length>=self.config.seq_len:
                selected_id = id
                break
        if selected_id == -1: #: no sequence of appropiate length for input
            raise SeqLengthTooHigh()             

        random_cut = random.randint(self.config.seq_len+1, length ) 
        dummy = df[df[case_id_column]==selected_id]

        return dummy.iloc[:random_cut]


    def check_input_uniqueness(self):
        """
        the input df must contain only one process. hence check if thereis one unique case_id 
        """
        return len(self.encoded_df[self.case_id_key].unique()) == 1
    

    def single_prediction_dataframe(self, df):
        """
        make one prediction given a dataframe. 
        preprocessor is in charge of doing the
        reading/importing from csv, xes, commandline, etc...
        """
        preprocessor = Preprocessing()
        preprocessor.import_event_log_dataframe(df, self.case_id_key, self.activity_key, self.timestamp_key)
        self.encoded_df= preprocessor.event_df 

        #: check case id uniqueness
        if not self.check_input_uniqueness():
            raise NotOneCaseId()
        self.current_case_id= self.encoded_df[self.case_id_key].sample(n = 1).values[0]
        return self.single_prediction()


    def single_prediction(self):
        """
        make one prediction given a partial process. 
        """

        #: sliding window
        step1= ATMDataset(self.config,self.encoded_df, self.case_id_key, self.timestamp_key, self.activity_key)
        #: just create one batch
        step2 = DataLoader(step1, batch_size=len(step1.time_seqs), shuffle=False, collate_fn=ATMDataset.to_features)

        #: TODO: refactor these lines here. no need for a for loop
        pred_times, pred_events = [], []
        for i, batch in enumerate(step2):   
            pred_time, pred_event = self.model.predict(batch, pm_active = True)
            pred_times.append(pred_time)
            pred_events.append(pred_event)
        time_pred = pred_times[-1][-1][-1]
        event_pred = pred_events[-1][-1] 
        
        return time_pred,  event_pred

    def jsonify_single(self, time_pred, event_pred, prob): 
        """
        note that we just save the
        probability of the last pair (time, event) in the path, 
        since the nn calculates lambda*(t) (see paper), which is 
        the probability of the last predicted event happening
        in the predicted time t. 
        """
        ans = {
            "timestamp": time_pred, 
            "event": self.config.activity_le.inverse_transform(event_pred),
            "prob": prob
        }
        json.dumps(ans)



    def get_differences(self):
        """
        calculates time differences. 
        """
        local = []
        for seq in self.recursive_time_seqs:
            seq = np.insert(seq, 0, seq[0])
            seq= np.diff(seq)
            local.append(seq)
        return np.array(local)


    def append_one_difference_array(self, lst):
        """
        appends one difference array to self.recursive_time_diffs 
        :param lst: list use for calculating the contiguous differences. 
        """
        #: extend the list by one element
        time = np.array([lst[0]]+ lst)
        #: get the differrnces between contiguous elements.
        time = np.diff(time)
        self.recursive_time_diffs= np.append(self.recursive_time_diffs, [time], axis = 0)
        
    def multiple_prediction_linear(self, depth, nonstop ,upper): 
        """
        this is a special case of multiple prediction
        where the degree= 1. we avoid backtracking and recursion for
        efficiency reasons.  
        """
        #: get the event and timestamp of the last event in the partial process.
        c_t =self.encoded_df[self.timestamp_key].iloc[-1]
        c_e =self.encoded_df[self.activity_key].iloc[-1]

        #: compute sliding window
        self.recursive_atm= ATMDataset(self.config, self.encoded_df, self.case_id_key, self.timestamp_key, self.activity_key, True)
        self.recursive_time_seqs = self.recursive_atm.time_seqs
        self.recursive_event_seqs = self.recursive_atm.event_seqs

        #: compute differences within each window
        self.recursive_time_diffs= self.get_differences()
        

        if nonstop: 
            #: in case that we run until end activity
            self.linear_iterative_predictor_non_stop(c_t, c_e, upper) 
        else:
            #: in case we run until a given depth
            self.linear_iterative_predictor(depth, c_t, c_e) 
        

    def linear_iterative_predictor_non_stop(self, start_time, start_event, upper): 
        """
        :param start_time: used to mark the start of the path 
        :param start_event: used to mark the start of the path
        :param upper: upper bound for the amount of iterations
        """
        c_t = start_time
        c_e = start_event
        path = [(c_t , (1,c_e))]
        i = 0
        #TODO: set upper bound as default to INFTY
        while not self.end_activities[c_e] and i<upper: #stop if end activity found or upper bound crossed
            p_t, p_events = self.get_sorted_wrapper() #get prediction
            p_pair = p_events[0] #get pair (prob, event)
            path.append((p_t[0], (p_pair[0], p_pair[1] ))) #save to path
            self.append_to_log(p_t[0], p_pair[1]) #update log for recusrive call
            c_t = p_t[0]
            c_e = p_pair[0]
            i+=1
        self.paths.append(path) #save generated path 
    

    def linear_iterative_predictor(self, depth, start_time, start_event): 
        """
        TODO: merge this fucntion with linear_iterative_predictor
        """
        c_t = start_time
        c_e = start_event
        path = [(c_t , (1,c_e))]
        for i in range(depth): 
            p_t, p_events = self.get_sorted_wrapper()
            p_pair = p_events[0]
            path.append((p_t[0], (p_pair[0], p_pair[1] )))
            self.append_to_log(p_t[0], p_pair[1])
        self.paths.append(path)

    def multiple_prediction(self, depth, degree): 
        """
        get a list of possible paths starting at the last 
        timestamp and event pair.  
        :param depth: how many steps in the future are to be predicted.
        :param degree: how many predictions on each step are to be considered.
        :param config: configuration used for the NN. required by ATM Dataset
        """
        c_t =self.encoded_df[self.timestamp_key].iloc[-1]
        c_e =self.encoded_df[self.activity_key].iloc[-1]

        #:load data, get windows
        self.recursive_atm= ATMDataset(self.config, self.encoded_df, self.case_id_key, self.timestamp_key, self.activity_key, True)
        self.recursive_time_seqs = self.recursive_atm.time_seqs
        self.recursive_event_seqs = self.recursive_atm.event_seqs

        #:get differences
        self.recursive_time_diffs= self.get_differences()

        #: compute paths
        self.backtracking_prediction_tree(c_t, c_e, 0,depth, degree,[(c_t, (1, c_e))]) 
        
        #: decode paths
        self.decode_paths()


    def backtracking_prediction_tree(self, c_t, c_e, c_d, depth, degree,current_path):
        """
        use backtracking to generate all the paths from the given 
        last timestamp and marker considering the input degree as a threshold 
        and the maximum depth for the generated tree.
        """
        if c_d >= depth: 
            # base case
            self.paths.append(list(current_path))
            return
        p_t, p_events = self.get_sorted_wrapper( )
        for p_e in p_events[:degree]:
            # filter branching degree ; the list is already sorted
            # therefore the "degree" most probable are taken
            self.append_to_log(p_t[0], p_e[1]) 
            current_path.append((p_t[0], p_e))
            self.backtracking_prediction_tree(p_t[0], p_e[1], c_d+1, depth, degree, list(current_path))    
            current_path.pop() 
            self.pop_from_log()
    
    def get_sorted_wrapper(self):
        
        #: check whether the number of rows in 
        # self.encoded_df <= seq_len. otherwise the
        # processed data by dataloader/atmdataset 
        # output empty lists... 
        if self.config.seq_len>= len(self.encoded_df):
            raise SeqLengthTooHigh()

        
        self.recursive_atm.event_seqs = self.recursive_event_seqs
        self.recursive_atm.time_seqs = self.recursive_time_seqs


        #: do not use dataloader for batch genertaion, too inefficient
        #: differnces are also computed in a smarter way, as well as windows.
        batch = ( torch.tensor(self.recursive_time_diffs,dtype=torch.float32), torch.tensor(self.recursive_event_seqs, dtype=torch.int64)) 


        pred_times, pred_events = [], []
        
        pred_time, pred_event = self.model.predict_sorted(batch)
        
        pred_times.append(pred_time)
        pred_events.append(pred_event)

        pred_times= pred_times[-1][-1] #we are only interested in the last one; unpack the batch
        pred_events = pred_events[-1][-1]

        return pred_times, pred_events


    def append_to_log(self,time, event): 
        """
        instead of calling ATMDataset and Dataloader on each iterative call 
        of the prediction generator, just append a window and a difference array
        to an existing list.  
        :param time: newly predicted timestamp
        :param event: newly predicted event 
        """
        last_time_seq = list(self.recursive_time_seqs[-1])
        last_event_seq = list(self.recursive_event_seqs[-1])
        new_time_seq = last_time_seq[1:]
        new_event_seq = last_event_seq[1:]
        new_time_seq.append(time)
        new_event_seq.append(event)
        self.recursive_time_seqs.append(new_time_seq)
        self.recursive_event_seqs.append(new_event_seq)
        self.append_one_difference_array(new_time_seq)
       
    def pop_from_log(self): 
        """
        used for backtracking to restore the old path
        """
        self.recursive_time_seqs.pop()
        self.recursive_event_seqs.pop()
        self.recursive_time_diffs= np.delete(self.recursive_time_diffs, len(self.recursive_time_diffs)-1, axis = 0) 

    def decode_paths(self): 
        """
        used for decoding the events and timestamps in the generated paths. 
        The timestamps are NOT decoded, since the predictions are TIMEDELTAS
        """
        #TODO: decode time considering time deltas

        self.decoded_paths = []
        for path in self.paths: 
            encoded_events = [event_index for _, (_, event_index) in path]
            encoded_events = list(map(int, encoded_events))
            decoded_events = self.config.activity_le.inverse_transform(encoded_events)

            decoded_path= [(time, (prob, event)) for (time, (prob, _)), event in zip(path, decoded_events) ]
            self.decoded_paths.append(decoded_path)
 
    def jsonify_paths(self): 
        """
        note that we just save the
        probability of the last pair (time, event) in the path, 
        since the nn calculates lambda*(t), which is 
        the probability of the last predicted event happening
        in the predicted time t. 
        """
        ans = {
            "paths": []
        }
        for path in self.decoded_paths :
            current_path = {
                'pairs':[], 
                'percentage': path[-1][1][1]
            }
            for time, (per, event) in path:
                current_path["pairs"].append(
                    {
                        "time": time, 
                        "event": event, 
                    }
                )
            ans["paths"].append(current_path)
        return json.dumps(ans)

    def multiple_prediction_dataframe(self, depth, degree, df, linear=False, non_stop = False, upper = 30):
        """
        make multiple predictions given a dataframe
        preprocessor is in charge of doing the
        reading/importing from csv, xes, commandline, etc...
        it is assumed that the event log contains only one case id.
        """
        preprocessor = Preprocessing()
        preprocessor.import_event_log_dataframe(df, self.case_id_key, self.activity_key, self.timestamp_key)
        
        self.encoded_df= preprocessor.event_df 
        self.current_case_id= self.encoded_df[self.case_id_key].sample(n = 1).values[0]
        if not linear: 
            """
            case backtracking needed; never called by prediction manager.  
            """
            self.multiple_prediction(depth, degree)
        else: 
            """
            the linear version is only being used for
            the iterative use of the multiple prediction generation.
            it mostly contains optimizations that are only 
            tought for the prediction manager. 
            """
            self.multiple_prediction_linear(depth, non_stop, upper)

