
import pm4py
from prediction_manager import PredictionManager
import random
import pandas as pd
from loggers import logger_generate_predictive_log, logger_multiple_prediction
from tqdm import tqdm

import time as tim

class ProcessModelManager:
    def __init__(self):
        self.predictive_df= None
        self.model = None 
        self.event_df = None
        self.case_activity_key =None 
        self.case_id_key =None 
        self.case_timestamp_key=None 
        self.case_id_le =None
        self.activity_le = None
        self.config = None


    def generate_predictive_log(self): 
        """
        this function creates the predictive event log in a dataframe. 
        we go over all case id's in the event log, and cut each
        sequence in the last three events. this is done for the following 
        reasons: 
        - bigger values for the cut generate longer prediction times
        hence slower performance. 
        - it doesnt make sense to make too long predictions in terms of
        sequence length, since the probability will always be significantly smaller.   
        after doing the predictions, the cut cases are extended with the predictions. 
        """
        st= tim.time()
        case_id_counts = self.event_df[self.case_id_key].value_counts()
        cuts = []
        self.predictive_df = {
            self.case_id_key:[],
            self.case_activity_key:[],
            self.case_timestamp_key:[]
        }
        self.predictive_df = pd.DataFrame(self.predictive_df)
        input_sequences = []
        cuts = {}
        for i, case_id in enumerate(case_id_counts.index):
            count = case_id_counts.loc[case_id]
            cut = random.randint(1, count)
            cut = count-min(3, cut)
            sequence = self.event_df[self.event_df[self.case_id_key]==case_id]  
            sequence = sequence.iloc[:cut]
            if len(sequence) <= self.config.seq_len: 
                continue
            cuts[case_id]= (count, cut, count-cut)
            input_sequences.append(sequence)
            self.predictive_df= pd.concat([self.predictive_df, sequence], ignore_index = True)
        
        #logger_generate_predictive_log.debug("cuts:")        
        #logger_generate_predictive_log.debug(cuts)        
        logger_generate_predictive_log.debug("no of input sequences:")        
        logger_generate_predictive_log.debug(len(input_sequences))        
        
        lenths = [len(seq) for seq in input_sequences]
        for i in lenths: 
            if i<=self.config.seq_len: 
                print("found too short sequence")
                return
        print("minimal length:")
        print(min(lenths))
        
        for i, sequence in enumerate(input_sequences) : 
            pm= PredictionManager()
            case_id = sequence[self.case_id_key].iloc[1]
            pm.model = self.model
            pm.case_id_le = self.case_id_le
            pm.activity_le = self.activity_le
            pm.seq_len = self.config.seq_len
            pm.multiple_prediction_dataframe(
                cuts[case_id][2],
                1,
                sequence, 
                self.case_activity_key  ,
                self.case_id_key  ,
                self.case_timestamp_key,
                self.config
            )

            prediction = pm.decoded_paths[0] #might break
            #logger_generate_predictive_log.debug("prediction len:")        
            #logger_generate_predictive_log.debug(len(prediction))
            #logger_generate_predictive_log.debug("prediction:")        
            extension = {
                self.case_id_key:[],
                self.case_activity_key:[],
                self.case_timestamp_key:[]
            }
            for time, (pred, event) in prediction: 
                extension[self.case_id_key] = case_id
                extension[self.case_activity_key]= event
                extension[self.case_timestamp_key]= time
            
            extension = pd.DataFrame(extension)
            #logger_generate_predictive_log.debug("extension:")        
            #logger_generate_predictive_log.debug(extension)        
            self.predictive_df= pd.concat([self.predictive_df, extension], ignore_index = True)

        logger_generate_predictive_log.debug("generated df:")        
        logger_generate_predictive_log.debug(self.predictive_df.head(20))        
    
        et= tim.time()
        logger_generate_predictive_log.debug("pred df creation duration:")        
        logger_generate_predictive_log.debug(et-st)



    def heuristic_miner(self):
        pass

    def inductive_miner(self):
        pass
    def alpha_miner(self):
        pass

    def correlation_miner(self):
        pass


    def prefix_tree_miner(self):
        pass