
import pm4py
from prediction_manager import PredictionManager
import random
import pandas as pd
from loggers import logger_generate_predictive_log, logger_multiple_prediction
from tqdm import tqdm

import time as tim

class ProcessModelManager:
    def __init__(self,event_df, model, config, case_activity_key, case_id_key, case_timestamp_key  ):
        #: it is assumed that the model already exists.
        self.model =model  #genertaed model
        self.case_activity_key = case_activity_key
        self.case_id_key = case_id_key
        self.case_timestamp_key= case_timestamp_key 
        self.config =config  #config of the nn
        self.event_df =event_df  #the df use for preprocessing and training 

        self.end_activities = {}
        self.predictive_df= None
        # variables to store PM model for further Conformance checking
        self.initial_marking = None
        self.final_marking = None
        self.petri_net = None



    def generate_predictive_log_random_cut_until_end(self, max_len, upper = 30): 
        """
        this function creates the predictive event log in a dataframe. 
        we go over all case id's in the event log, and cut each
        sequence in a random index. 
        an upper bound can be selected for the max sequence length
        to reduce runtime.
        after this, we try to complet the cut trace until a 
        final activity is predicted.  
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
            sequence = self.event_df[self.event_df[self.case_id_key]==case_id]  
            if count<=self.config.seq_len or max_len>100:
                continue
            cut = random.randint(self.config.seq_len+1, count)
            sequence = sequence.iloc[:cut]
            cuts[case_id]= (count, cut, count-cut)
            input_sequences.append(sequence)
            self.predictive_df= pd.concat([self.predictive_df, sequence], ignore_index = True)
        
        
        lenths = [len(seq) for seq in input_sequences]
        for i in lenths: 
            if i<=self.config.seq_len: 
                print("found too short sequence")
                return
        
        for i, sequence in enumerate(tqdm(input_sequences)): 
            pm= PredictionManager(
                self.model, 
                self.case_id_key, 
                self.case_activity_key, 
                self.case_timestamp_key, 
                self.config
            )
            pm.end_activities = self.end_activities
            case_id = sequence[self.case_id_key].iloc[1]
            pm.model = self.model
            pm.case_id_le = self.config.case_id_le
            pm.activity_le = self.config.activity_le
            pm.seq_len = self.config.seq_len
            pm.multiple_prediction_dataframe(
                cuts[case_id][2],
                1,
                sequence, 
                linear = True, 
                non_stop = True ,
                upper = upper 
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
                extension[self.case_id_key] = [case_id]
                extension[self.case_activity_key]= [event]
                extension[self.case_timestamp_key]= [time]
            
            #logger_generate_predictive_log.debug("extension:")        
            #logger_generate_predictive_log.debug(extension)        
            extension = pd.DataFrame(extension)
            #logger_generate_predictive_log.debug("extension:")        
            #logger_generate_predictive_log.debug(extension)        
            self.predictive_df= pd.concat([self.predictive_df, extension], ignore_index = True)

        logger_generate_predictive_log.debug("generated df:")        
        logger_generate_predictive_log.debug(self.predictive_df.head(20))        
    
        et= tim.time()
        logger_generate_predictive_log.debug("pred df creation duration:")        
        logger_generate_predictive_log.debug(et-st)






    def generate_predictive_log_random_cut(self, upper_bound): 
        """
        this function creates the predictive event log in a dataframe. 
        we go over all case id's in the event log, and cut each
        sequence in some random index bigger than seq length. 
        for performance reasons the user can choose an  upper bound
        for the selected traces. 
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
            sequence = self.event_df[self.event_df[self.case_id_key]==case_id]  
            if count<=self.config.seq_len or count>upper_bound:
                continue
            cut = random.randint(self.config.seq_len+1, count)
            sequence = sequence.iloc[:cut]
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
        
        for i, sequence in enumerate(tqdm(input_sequences)): 
            pm= PredictionManager(
                self.model, 
                self.case_id_key, 
                self.case_activity_key, 
                self.case_timestamp_key, 
                self.config
            )
            case_id = sequence[self.case_id_key].iloc[1]
            pm.model = self.model
            pm.case_id_le = self.config.case_id_le
            pm.activity_le = self.config.activity_le
            pm.seq_len = self.config.seq_len
            pm.multiple_prediction_dataframe(
                cuts[case_id][2],
                1,
                sequence, 
                linear = True  
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
                extension[self.case_id_key] = [case_id]
                extension[self.case_activity_key]= [event]
                extension[self.case_timestamp_key]= [time]
            
            #logger_generate_predictive_log.debug("extension:")        
            #logger_generate_predictive_log.debug(extension)        
            extension = pd.DataFrame(extension)
            #logger_generate_predictive_log.debug("extension:")        
            #logger_generate_predictive_log.debug(extension)        
            self.predictive_df= pd.concat([self.predictive_df, extension], ignore_index = True)

        logger_generate_predictive_log.debug("generated df:")        
        logger_generate_predictive_log.debug(self.predictive_df.head(20))        
    
        et= tim.time()
        logger_generate_predictive_log.debug("pred df creation duration:")        
        logger_generate_predictive_log.debug(et-st)







    def generate_predictive_log_tail_cut(self): 
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
        
        for i, sequence in enumerate(tqdm(input_sequences)): 
            pm= PredictionManager(
                model = self.model, 
                case_id_key= self.case_id_key, 
                activity_key= self.case_activity_key,
                timestamp_key=self.case_timestamp_key,
                config = self.config
            )
            case_id = sequence[self.case_id_key].iloc[1]
            pm.model = self.model
            pm.case_id_le = self.config.case_id_le
            pm.activity_le = self.config.activity_le
            pm.seq_len = self.config.seq_len
            pm.multiple_prediction_dataframe(
                cuts[case_id][2],
                1,
                sequence, 
                linear = True
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
                extension[self.case_id_key] = [case_id]
                extension[self.case_activity_key]= [event]
                extension[self.case_timestamp_key]= [time]
            
            extension = pd.DataFrame(extension)
            #logger_generate_predictive_log.debug("extension:")        
            #logger_generate_predictive_log.debug(extension)        
            self.predictive_df= pd.concat([self.predictive_df, extension], ignore_index = True)

        logger_generate_predictive_log.debug("generated df:")        
        logger_generate_predictive_log.debug(self.predictive_df.head(20))        
    
        et= tim.time()
        logger_generate_predictive_log.debug("pred df creation duration:")        
        logger_generate_predictive_log.debug(et-st)

    def decode_df(self):

        logger_generate_predictive_log.debug("numbers of nan before decode")        
        logger_generate_predictive_log.debug(self.predictive_df.isna().sum())        
        logger_generate_predictive_log.debug(self.predictive_df.head(20))        
        logger_generate_predictive_log.debug(self.predictive_df.tail(20))        
        self.predictive_df[self.case_activity_key] = self.predictive_df[self.case_activity_key].astype("str")
        self.predictive_df[self.case_id_key] = self.predictive_df[self.case_activity_key].astype("str")
        self.predictive_df[self.case_timestamp_key] = self.predictive_df[self.case_timestamp_key]*(10**self.config.exponent)
        self.predictive_df[self.case_timestamp_key] = self.predictive_df[self.case_timestamp_key].astype("datetime64[ns]")
        """
        TODO: all time predictions >=1 ARE NOT CORRECT.  (not convertible)
        - either catch them an raise an error
        - drop them 
        - or find another solution
        """

        logger_generate_predictive_log.debug("number of nan after decode")        
        logger_generate_predictive_log.debug(self.predictive_df.isna().sum())        
        self.predictive_df = self.predictive_df.dropna() #just for now
        self.predictive_df.to_csv("logs/predicted_df")
        logger_generate_predictive_log.debug(self.predictive_df.head(20))        
        logger_generate_predictive_log.debug(self.predictive_df.tail(20))        


    def heuristic_miner(self):
        self.decode_df()
        self.petri_net, self.initial_marking, self.final_marking = pm4py.discover_petri_net_heuristics(self.predictive_df, activity_key=self.case_activity_key,timestamp_key=self.case_timestamp_key,case_id_key= self.case_id_key)
        #pm4py.view_petri_net(self.petri_net, self.initial_marking, self.final_marking, format='svg')
        pm4py.view_petri_net(self.petri_net, format='svg')

    def inductive_miner(self):
        self.decode_df()
        self.petri_net, self.initial_marking, self.final_marking = pm4py.discover_petri_net_inductive(self.predictive_df, self.case_activity_key,self.case_timestamp_key, self.case_id_key)
        pm4py.view_petri_net(self.petri_net, self.initial_marking, self.final_marking, format='svg')

    def alpha_miner(self):
        self.decode_df()
        self.petri_net, self.initial_marking, self.final_marking = pm4py.discover_petri_net_alpha(self.predictive_df, self.case_activity_key,self.case_timestamp_key, self.case_id_key)
        pm4py.view_petri_net(self.petri_net, self.initial_marking, self.final_marking, format='svg')

    def prefix_tree_miner(self):
        self.decode_df()
        self.petri_net, self.initial_marking, self.final_marking = pm4py.discover_prefix_tree(self.predictive_df, self.case_activity_key,self.case_timestamp_key, self.case_id_key)
        pm4py.view_petri_net(self.petri_net, self.initial_marking, self.final_marking, format='svg')

        '''
        might be irrelevant as we require to always have a case_identifier in the log input 
        -> correlation miner only useful if we do not have or know the case_identifier
        def correlation_miner(self):
        pass
        '''

    def conformance_checking_token_based_replay(self):
        replayed_traces = pm4py.conformance_diagnostics_token_based_replay(self.predictive_df, self.petri_net, self.initial_marking, self.final_marking)
