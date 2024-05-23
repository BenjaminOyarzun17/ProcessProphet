from loggers import logger_get_dummy_process,logger_single_prediction , logger_multiple_prediction
from exceptions import ProcessTooShort, SeqLengthTooHigh
from preprocessing import Preprocessing
from ERPP_RMTPP_torch import * 
from torch.utils.data import DataLoader
import pandas as pd
import pprint 
import json




class PredictionManager: 
    def __init__(self):
        #TODO: it might be more convenient to store a config object 
        # so that not so much copying is necessary
        self.model = None
        self.input_df = None
        self.encoded_df = None
        self.case_id_key = None
        self.activity_key = None
        self.timestamp_key = None
        self.current_case_id = None
        self.paths = []
        self.decoded_paths= []
        self.case_id_le = None
        self.activity_le = None
        self.seq_len = None





    def get_dummy_process(self, df, case_id_column):
        """
        just used for testing
        """
        random_case_id = df[case_id_column].sample(n = 1).values[0]
        dummy = df[df[case_id_column]==random_case_id]
        n_rows = dummy.shape[0]
        if n_rows == 1:
            raise ProcessTooShort()
        logger_get_dummy_process.debug(dummy) 
        logger_get_dummy_process.debug(dummy.iloc[:10]) 

        return dummy.iloc[:n_rows-1]



    

    def single_prediction_dataframe(self, df, case_id, activity_key, timestamp_key, config):
        """
        make one prediction given a dataframe. 
        preprocessor is in charge of doing the
        reading/importing from csv, xes, commandline, etc...
        """
        preprocessor = Preprocessing()
        preprocessor.import_event_log_dataframe(df, case_id, activity_key, timestamp_key )
        encoded_df= preprocessor.event_df 
        self.encoded_df= preprocessor.event_df 
        self.activity_key = activity_key
        self.case_id_key = case_id
        self.timestamp_key = timestamp_key
        self.current_case_id= self.encoded_df[self.case_id_key].sample(n = 1).values[0]
        self.single_prediction(config )


    def single_prediction(self,  config ):
        """
        make one prediction 
        """
        step1= ATMDataset(config,self.encoded_df, self.case_id_key, self.timestamp_key, self.activity_key)
        #: batch size set to one to have one sample per batch.
        step2 = DataLoader(step1, batch_size=1, shuffle=False, collate_fn=ATMDataset.to_features)
        pred_times, pred_events = [], []
        for i, batch in enumerate(step2):   
            pred_time, pred_event = self.model.predict(batch)
            pred_times.append(pred_time)
            pred_events.append(pred_event)
        logger_single_prediction.debug("predicted time:")
        logger_single_prediction.debug(pred_times)
        logger_single_prediction.debug("predicted event:")
        logger_single_prediction.debug(pred_events)
        return pred_times, pred_events



    def multiple_prediction(self, depth, degree,  config): 
        """
        get a list of possible paths starting at the last 
        timestamp and event pair.  
        :param depth: how many steps in the future are to be predicted.
        :param degree: how many predictions on each step are to be considered.
        :param config: TODO 
        """
        logger_multiple_prediction.debug("dummies:")
        logger_multiple_prediction.debug(self.encoded_df)
        c_t =self.encoded_df[self.timestamp_key].iloc[-1]
        c_e =self.encoded_df[self.activity_key].iloc[-1]
        self.backtracking_prediction_tree(c_t, c_e, 0,depth, degree,[(c_t, (1, c_e))],   config) 
        self.decode_paths()
        pp = pprint.PrettyPrinter(indent = 4)
        pp.pprint(self.decoded_paths)
        logger_multiple_prediction.debug("paths:")
        logger_multiple_prediction.debug(self.paths)

    def decode_paths(self): #TODO: decode time
        self.decoded_paths = []
        for path in self.paths: 

            encoded_events = [event_index for _, (_, event_index) in path]
            encoded_events = list(map(int, encoded_events))
            print(encoded_events)
            decoded_events = self.activity_le.inverse_transform(encoded_events)
            decoded_path= [(time, (prob, event)) for (time, (prob, _)), event in zip(path, decoded_events) ]
            print(decoded_path)



            self.decoded_paths.append(decoded_path)
        
    def jsonify_paths(self): 
        ans = {
            "paths": []
        }
        for time, (per, event) in self.decoded_paths :
            ans["paths"].append(
                {
                    "time": time, 
                    "event": event, 
                    "percentage": per
                }
            )

        return json.dumps(ans)


    def backtracking_prediction_tree(self, c_t, c_e, c_d, depth, degree,current_path , config):
        """
        use backtracking to generate all the paths from the given 
        last timestamp and marker.
        """
        if c_d >= depth: 
            self.paths.append(list(current_path))
            return
        p_t, p_events = self.get_sorted_wrapper( config )
        for p_e in p_events[:degree]: 
            self.append_to_log(p_t[0], p_e[1]) 
            current_path.append((p_t, p_e))
            self.backtracking_prediction_tree(p_t[0], p_e[1], c_d+1, depth, degree, list(current_path), config)    
            current_path.pop() 
            self.pop_from_log()
    
    def get_sorted_wrapper(self, config ):
        
        #: check whether the number of rows in 
        # self.encoded_df <= seq_len. otherwise the
        # processed data by dataloader/atmdataset 
        # output empty lists... 
        if self.seq_len>= len(self.encoded_df):
            raise SeqLengthTooHigh()

        step1= ATMDataset(config, self.encoded_df, self.case_id_key, self.timestamp_key, self.activity_key)
        #: batch size set to one to have one sample per batch.
        step2 = DataLoader(step1, batch_size=1, shuffle=False, collate_fn=ATMDataset.to_features)

        pred_times, pred_events = [], []
        for i, batch in enumerate(step2):   
            logger_multiple_prediction.debug("batch:")
            logger_multiple_prediction.debug(pred_times)
            pred_time, pred_event = self.model.predict_sorted(batch)
            pred_times.append(pred_time)
            pred_events.append(pred_event)
        logger_multiple_prediction.debug("predicted time:")
        logger_multiple_prediction.debug(pred_times)
        logger_multiple_prediction.debug("predicted event:")
        logger_multiple_prediction.debug(pred_events)

        pred_times= pred_times[-1][-1] #we are only interested in the last one; unpack the batch
        pred_events = pred_events[-1][-1]

        logger_multiple_prediction.debug("predicted time:")
        logger_multiple_prediction.debug(pred_times)
        logger_multiple_prediction.debug("predicted event:")
        logger_multiple_prediction.debug(pred_events)

        return pred_times, pred_events
    def append_to_log(self,time, event): 
        extra = {
                self.case_id_key:self.current_case_id, 
                self.timestamp_key:time, 
                self.activity_key:event, 
            }
        self.encoded_df.loc[len(self.encoded_df)] = extra

    def pop_from_log(self): 
        self.encoded_df = self.encoded_df.iloc[:-1]

 

    def multiple_prediction_dataframe(self, depth, degree, df, case_id, activity_key, timestamp_key, config):
        """
        make multiple predictions given a dataframe
        preprocessor is in charge of doing the
        reading/importing from csv, xes, commandline, etc...
        """
        preprocessor = Preprocessing()
        preprocessor.import_event_log_dataframe(df, case_id, activity_key, timestamp_key )
        self.activity_key = activity_key
        self.case_id_key = case_id
        self.timestamp_key = timestamp_key
        self.encoded_df= preprocessor.event_df 
        self.current_case_id= self.encoded_df[self.case_id_key].sample(n = 1).values[0]
        self.multiple_prediction(depth, degree,  config )

