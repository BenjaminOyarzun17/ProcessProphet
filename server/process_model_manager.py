
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
        """
        initialize variabels for predictive log generator
        """
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
        """
        cuts the each sequence contained in input sequences at random indices. 
        :param cuts: the cut index and cut length are preserved
        :param case_id_counts: number of rows for each case_id
        :max_len: max length that the input sequence can have. can be set to improve runtime 
        TODO: allow INF for max_len
        :param input_sequences: list of sequences to be cut. 
        """
        for i, case_id in enumerate(case_id_counts.index):
            count = case_id_counts.loc[case_id]
            sequence = self.event_df[self.event_df[self.case_id_key]==case_id]  
            if count<=self.config.seq_len or count>max_len:
                continue
            cut = random.randint(self.config.seq_len+1, count)
            sequence = sequence.iloc[:cut]
            cuts[case_id]= (count, cut, count-cut)
            input_sequences.append(sequence)
            self.predictive_df= pd.concat([self.predictive_df, sequence], ignore_index = True)
        return case_id_counts, cuts, input_sequences 

    def fill_up_log(self, upper , non_stop , random_cuts , cut_length , input_sequences, cuts):
        """
        do the predictions for each cut sequence and extend the event log so that 
        it now constains the predictions. 
        """
        #: initialize prediction manager.
        pm= PredictionManager(
            self.model, 
            self.case_id_key, 
            self.case_activity_key, 
            self.case_timestamp_key, 
            self.config
        )
        pm.end_activities = self.end_activities

        for i, sequence in enumerate(tqdm(input_sequences)): 
            case_id = sequence[self.case_id_key].iloc[1]
            #: do the predictions in the corresponding mode
            pm.multiple_prediction_dataframe(
                cuts[case_id][2],
                1,
                sequence, 
                linear = True ,  
                non_stop=non_stop,
                upper = upper, 
            )
            prediction = pm.decoded_paths[0] 
            extension = {
                self.case_id_key:[],
                self.case_activity_key:[],
                self.case_timestamp_key:[]
            }
            #: arrange the predictions in the extension dictionary
            for time, (pred, event) in prediction: 
                extension[self.case_id_key] = [case_id]
                extension[self.case_activity_key]= [event]
                extension[self.case_timestamp_key]= [time]
            
            #: transform extension to dtaframe and extend the predictive df now with the predictions
            extension = pd.DataFrame(extension)
            self.predictive_df= pd.concat([self.predictive_df, extension], ignore_index = True)



    def generate_predictive_log(self, new_log_path, max_len= 15, upper = 30, non_stop = False, random_cuts = False, cut_length = 0): 
        """
        generates a predictive log. each process is cut at some given index, and the model is used to 
        reconstruct the rest of the process. there are so far three possible modi for cutting and prediction generation:  
        - for tail cuts: set cut_length value and set random_cuts to false
        - for random cuts with cut memory: random_cuts to true and non_stop to false
        - for random cuts nonstop: random_cuts to true and non_stop totrue 

        :param max len: max length for the cut sequences ie max sequence input size length.
        :param upper:  upperbound for the non stop random cutter ie how long to run before reaching end state. 
        :param non_stop: must be set to true if the predictions are done until reaching final marking.
        :param random_cuts: set to true to cut in random indices. 
        :param cut_length: in case of cutting fix tail lengths, select the tail length to cut for all sequences.
        :param upper: upper bound for how many iterations a non stop iterative predictor should run.
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
        self.predictive_df.to_csv(new_log_path, sep = ",")


    def check_too_short(self, sequences):
        lenths = [len(seq) for seq in sequences]
        for i in lenths: 
            if i<=self.config.seq_len: 
                print("found too short sequence")
                raise CutTooLarge()

    def tail_cutter(self, case_id_counts, cut_length, cuts, input_sequences):
        """
        cut sequences cut_length steps from the tail.
        :param cut_length: how many steps to cut from the tail of each sequence. 
        :param case_id_counts: number of steps on each case_id
        :param input_sequences: list of sequences to be cut. 
        """
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

    def import_predictive_df(self, path):
        self.predictive_df = pd.read_csv(path, sep = ",")


    def heuristic_miner(self,path,  dependency_threshold=0.5, and_threshold=0.65, loop_two_threshold=0.5):
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
        #pm4py.view_petri_net(self.petri_net, format='svg')
        pm4py.write_pnml(self.petri_net,self.initial_marking, self.final_marking, file_path=path)


    def inductive_miner(self, path,   noise_threshold=0):
        self.decode_df()
        self.petri_net, self.initial_marking, self.final_marking = pm4py.discover_petri_net_inductive(
            self.predictive_df,
            noise_threshold, 
            self.case_activity_key,
            self.case_timestamp_key,
            self.case_id_key
        )
        #pm4py.view_petri_net(self.petri_net, self.initial_marking, self.final_marking, format='svg')
        pm4py.write_pnml(self.petri_net,self.initial_marking, self.final_marking, file_path=path)

    def alpha_miner(self, path):
        self.decode_df()
        self.petri_net, self.initial_marking, self.final_marking = pm4py.discover_petri_net_alpha(
            self.predictive_df,
            self.case_activity_key,
            self.case_timestamp_key, 
            self.case_id_key
        )
        #pm4py.view_petri_net(self.petri_net, self.initial_marking, self.final_marking, format='svg')
        pm4py.write_pnml(self.petri_net,self.initial_marking, self.final_marking , file_path=path)

    def prefix_tree_miner(self, path):
        self.decode_df()
        self.petri_net, self.initial_marking, self.final_marking = pm4py.discover_prefix_tree(
            self.predictive_df,
            self.case_activity_key,
            self.case_timestamp_key,
            self.case_id_key
        )
        #pm4py.view_petri_net(self.petri_net, self.initial_marking, self.final_marking, format='svg')
        pm4py.write_pnml(self.petri_net,self.initial_marking, self.final_marking , file_path=path)

        '''
        might be irrelevant as we require to always have a case_identifier in the log input 
        -> correlation miner only useful if we do not have or know the case_identifier
        def correlation_miner(self):
        pass
        '''

    def conformance_checking_token_based_replay(self):
        replayed_traces = pm4py.conformance_diagnostics_token_based_replay(self.predictive_df, self.petri_net, self.initial_marking, self.final_marking)
