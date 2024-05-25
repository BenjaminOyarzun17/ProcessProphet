
import pm4py
from prediction_manager import PredictionManager
import random
import pandas as pd
from loggers import logger_generate_predictive_log, logger_multiple_prediction
from tqdm import tqdm
from exceptions import CutTooLarge, CutLengthZero

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

    def initialize_variables(self):

        case_id_counts = self.event_df[self.case_id_key].value_counts()
        cuts = []
        self.predictive_df = {
            self.case_id_key:[],
            self.case_activity_key:[],
            self.case_timestamp_key:[]
        }
        input_sequences = []
        cuts = {}
        return case_id_counts, cuts, input_sequences, cuts

    def random_cutter(self, case_id_counts, max_len, cuts, input_sequences):
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
        return case_id_counts, cuts, input_sequences 

    def fill_up_log(self, upper , non_stop , random_cuts , cut_length , input_sequences, cuts):

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
            pm.end_activities = self.end_activities
            pm.multiple_prediction_dataframe(
                cuts[case_id][2],
                1,
                sequence, 
                linear = True ,  
                non_stop=non_stop,
                upper = upper, 
            )
            prediction = pm.decoded_paths[0] #might break
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



    def generate_predictive_log(self, max_len= 15, upper = 30, non_stop = False, random_cuts = False, cut_length = 0): 
        """
       

        max len: max length for the cut sequences ie max sequence input size length.
        upper:  upperbound for the non stop random cutter ie how long to run before reaching end state. 

        for tail cuts: set cut_length value and set random_cuts to false
        for random cuts with cut memory: random_cuts to true and non_stop to false
        for random cuts nonstop: random_cuts to true and non_stop totrue 
        
        """
        st= tim.time()
        
        case_id_counts, cuts, input_sequences, cuts = self.initialize_variables()

        self.predictive_df = pd.DataFrame(self.predictive_df)
        if random_cuts: 
            case_id_counts,cuts, input_sequences= self.random_cutter(case_id_counts, max_len,cuts,  input_sequences)
        else: 
            if cut_length ==0: 
                raise CutLengthZero()
            case_id_counts,cuts, input_sequences= self.tail_cutter(case_id_counts, cut_length,cuts,  input_sequences)
            
        
        self.check_too_short(input_sequences)

        self.fill_up_log( upper , non_stop , random_cuts , cut_length , input_sequences, cuts)

        logger_generate_predictive_log.debug("generated df:")        
        logger_generate_predictive_log.debug(self.predictive_df.head(20))        
    
        et= tim.time()
        logger_generate_predictive_log.debug("pred df creation duration:")        
        logger_generate_predictive_log.debug(et-st)



    def check_too_short(self, sequences):
        lenths = [len(seq) for seq in sequences]
        for i in lenths: 
            if i<=self.config.seq_len: 
                print("found too short sequence")
                raise CutTooLarge()

    def tail_cutter(self, case_id_counts, cut_length, cuts, input_sequences):

        for i, case_id in enumerate(case_id_counts.index):
            count = case_id_counts.loc[case_id]
            cut = random.randint(1, count)
            cut = count-min(cut_length, cut)
            sequence = self.event_df[self.event_df[self.case_id_key]==case_id]  
            sequence = sequence.iloc[:cut]
            if len(sequence) <= self.config.seq_len: 
                continue
            cuts[case_id]= (count, cut, count-cut)
            input_sequences.append(sequence)

            sequence = self.decode_sequence(sequence)
            self.predictive_df= pd.concat([self.predictive_df, sequence], ignore_index = True)

        return case_id_counts, cuts, input_sequences 

    

    def decode_sequence(self, sequence):
        sequence[self.case_activity_key] = self.config.activity_le.inverse_transform(sequence[self.case_activity_key].astype(int))
        return sequence


    def handle_nat(self, group):
        last_valid_idx = group[self.case_timestamp_key].last_valid_index()
        if last_valid_idx is None:
            return group
        last_valid_timestamp= group.loc[last_valid_idx, self.case_timestamp_key]
        
        nat_indices = group.index[group[self.case_timestamp_key].isna()]
        for i, idx in enumerate(nat_indices):
            group.at[idx, self.case_timestamp_key] = last_valid_timestamp+ pd.Timedelta(days=i + 1)
    
        return group     





    def decode_df(self):

        logger_generate_predictive_log.debug("numbers of nan before decode")        
        logger_generate_predictive_log.debug(self.predictive_df.isna().sum())        
        logger_generate_predictive_log.debug(self.predictive_df.head(20))        
        logger_generate_predictive_log.debug(self.predictive_df.tail(20))        
        self.predictive_df[self.case_activity_key] = self.predictive_df[self.case_activity_key].astype("str")
        self.predictive_df[self.case_id_key] = self.predictive_df[self.case_activity_key].astype("str")
        #: note that this operation is lossy and might generate NaT. 
        self.predictive_df[self.case_timestamp_key] = self.predictive_df[self.case_timestamp_key]*(10**self.config.exponent)
        self.predictive_df[self.case_timestamp_key] = self.predictive_df[self.case_timestamp_key].astype("datetime64[ns]")

        self.predictive_df= self.predictive_df.groupby(self.case_id_key, group_keys=False).apply(self.handle_nat)

        self.predictive_df.to_csv("logs/predicted_df")


    def heuristic_miner(self, dependency_threshold=0.5, and_threshold=0.65, loop_two_threshold=0.5):
        self.decode_df()
        self.petri_net, self.initial_marking, self.final_marking = pm4py.discover_petri_net_heuristics(
            self.predictive_df,
            dependency_threshold, 
            and_threshold, 
            loop_two_threshold, 
            activity_key=self.case_activity_key,
            timestamp_key=self.case_timestamp_key,
            case_id_key= self.case_id_key
        )
        #pm4py.view_petri_net(self.petri_net, self.initial_marking, self.final_marking, format='svg')
        pm4py.view_petri_net(self.petri_net, format='svg')

    def inductive_miner(self,  noise_threshold=0):
        self.decode_df()
        self.petri_net, self.initial_marking, self.final_marking = pm4py.discover_petri_net_inductive(
            self.predictive_df,
            noise_threshold, 
            self.case_activity_key,
            self.case_timestamp_key,
            self.case_id_key
        )
        pm4py.view_petri_net(self.petri_net, self.initial_marking, self.final_marking, format='svg')

    def alpha_miner(self):
        self.decode_df()
        self.petri_net, self.initial_marking, self.final_marking = pm4py.discover_petri_net_alpha(
            self.predictive_df,
            self.case_activity_key,
            self.case_timestamp_key, 
            self.case_id_key
        )
        pm4py.view_petri_net(self.petri_net, self.initial_marking, self.final_marking, format='svg')

    def prefix_tree_miner(self):
        self.decode_df()
        self.petri_net, self.initial_marking, self.final_marking = pm4py.discover_prefix_tree(
            self.predictive_df,
            self.case_activity_key,
            self.case_timestamp_key,
            self.case_id_key
        )
        pm4py.view_petri_net(self.petri_net, self.initial_marking, self.final_marking, format='svg')

        '''
        might be irrelevant as we require to always have a case_identifier in the log input 
        -> correlation miner only useful if we do not have or know the case_identifier
        def correlation_miner(self):
        pass
        '''

    def conformance_checking_token_based_replay(self):
        replayed_traces = pm4py.conformance_diagnostics_token_based_replay(self.predictive_df, self.petri_net, self.initial_marking, self.final_marking)
