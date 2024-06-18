"""
This module implements all necessary functions for conformance checking and fitness analysis.

Functions:
    - cut_event_log_tail: Cuts each case in the event log from the tail.
    - cut_event_log_random: Cuts each case in the event log at random indices.
    - reconstruct_event_log: Reconstructs the event log using the prediction manager.
    - process_mining: Applies a process mining algorithm to the reconstructed event log.
    - conformance_checking_token_based: Performs token-based conformance checking on the reconstructed event log.
    - conformance_checking_alignment_based: Performs alignment-based conformance checking on the reconstructed event log.
    - import_petri_net: Imports a Petri net.
    - export_petri_net: Exports a Petri net.
    - decode_predictions: Decodes the predictions in the event log.

This module allows for the analysis of fitness by cutting the event log, reconstructing it using predictions, and applying process mining and conformance checking algorithms.
"""
import pm4py
from server  import prediction_manager
from server  import loggers
from server import exceptions

import random
from pm4py.algo.evaluation.replay_fitness import algorithm as replay_fitness
import pandas as pd
from tqdm import tqdm

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
        self.unencoded_df = None #used for conformance checking
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

    """
    def alignment_input(self, cut_length, total_traces): 
        - cuts `cut_length` elements from the tail of each of the sampled `total_traces` traces. 
        - the cut out subsequences are saved in a dataframe.
        - the partial sequence (what remained after cutting) is passed as input for the pred. model
        and `cut_length` predictions are done for each trace. 
        - with this we can later compute alignments.  
        this conformance chcking method should better quantify how good the model is at reconstructing 
        past behaviour for each trace individually, without considering the part that was using for training.  
    """



    def tail_cutter(self, case_id_counts, cut_length, cuts, input_sequences):
        """
        cut sequences cut_length steps from the tail.
        :param cut_length: how many steps to cut from the tail of each sequence. 
        :param case_id_counts: number of steps on each case_id
        :param input_sequences: list of sequences to be cut. 

        Side effect: the predictive_df is extended with the cut sequences.
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

            self.predictive_df= pd.concat([self.predictive_df, sequence], ignore_index = True)

        return case_id_counts, cuts, input_sequences 

    def random_cutter(self, case_id_counts, max_len, cuts, input_sequences):
        """
        Cuts each sequence contained in input_sequences at random indices.
        
        Args:
            cuts (dict): The cut index and cut length are preserved.
            case_id_counts (pd.Series): Number of rows for each case_id.
            max_len (int): Max length that the input sequence can have. Can be set to improve runtime.
                   TODO: allow INF for max_len.
            input_sequences (list): List of sequences to be cut.
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
        pm=prediction_manager.PredictionManager(
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
                linear = True,  
                non_stop=non_stop,
                upper = upper,
            )
            prediction = pm.paths[0] 
            extension = {
                self.case_id_key:[],
                self.case_activity_key:[],
                self.case_timestamp_key:[]
            }
            #: arrange the predictions in the extension dictionary
            # the first prediction is not used because it is just information 
            # use for knowing what was the last timestamp recorded
            for time, (pred, event) in prediction[1:]:
                extension[self.case_id_key] = [case_id]
                extension[self.case_activity_key]= [event]
                extension[self.case_timestamp_key]= [time]
            
            extension = pd.DataFrame(extension)
            #: here compute cumulative sum of times + last timestamp recorded
            extension[self.case_timestamp_key] = extension[self.case_timestamp_key]*self.config.exponent
            extension[self.case_timestamp_key] = extension[self.case_timestamp_key].cumsum()
            extension[self.case_timestamp_key] = extension[self.case_timestamp_key] + prediction[0][0]
            
            #: transform extension to dtaframe and extend the predictive df now with the predictions
            self.predictive_df= pd.concat([self.predictive_df, extension], ignore_index = True)

        self.predictive_df=  self.predictive_df.sort_values(by=[self.case_id_key, self.case_timestamp_key])
        self.predictive_df = self.decode_df(self.predictive_df)

    def generate_predictive_log(self, new_log_path, max_len= 15, upper = 30, non_stop = False, random_cuts = False, cut_length = 0): 
        """
        generates a predictive log. each process is cut at some given index, and the model is used to 
        reconstruct the rest of the process. there are so far three possible modi for cutting and prediction generation:  
        - for tail cuts: set cut_length value and set random_cuts to false
        - for random cuts with cut memory: random_cuts to true and non_stop to false
        - for random cuts nonstop: random_cuts to true and non_stop totrue 

        Args:
            max len: max length for the cut sequences ie max sequence input size length.
            upper:  upperbound for the non stop random cutter ie how long to run before reaching end state. 
            non_stop: must be set to true if the predictions are done until reaching final marking.
            random_cuts: set to true to cut in random indices. 
            cut_length: in case of cutting fix tail lengths, select the tail length to cut for all sequences.
            upper: upper bound for how many iterations a non stop iterative predictor should run.
        """
        
        case_id_counts, cuts, input_sequences, cuts = self.initialize_variables()

        self.predictive_df = pd.DataFrame(self.predictive_df)
        if random_cuts: 
            case_id_counts,cuts, input_sequences= self.random_cutter(case_id_counts, max_len,cuts,  input_sequences)
        else: 
            if cut_length ==0: 
                raise exceptions.CutLengthZero()
            case_id_counts,cuts, input_sequences= self.tail_cutter(case_id_counts, cut_length,cuts,  input_sequences)
            
        
        self.check_too_short(input_sequences)

        self.fill_up_log( upper , non_stop , random_cuts , cut_length , input_sequences, cuts)
    
        self.predictive_df.to_csv(new_log_path, sep = ",")

    def check_too_short(self, sequences):
        lenths = [len(seq) for seq in sequences]
        for i in lenths: 
            if i<=self.config.seq_len: 
                print("found too short sequence")
                raise exceptions.CutTooLarge()

    def decode_sequence(self, sequence):
        """
        decodes the input sequence that contains a df.  
        :return: sequence that has been decoded. 
        """
        sequence[self.case_activity_key] = self.config.activity_le.inverse_transform(sequence[self.case_activity_key].astype(int))
        return sequence

    def handle_nat(self, group):
        """
        the inverse transformation for timestamps is a lossy transformation and might lead to NaT entries. 
        a timedelta of k second's with respect to the last valid timestamp is set as a timestamp value for 
        the kth NaT entry.
        :param group: a group in the predictive df that contains only one case id. 
        :return: the same group now with valid timestamps
        """
        last_valid_idx = group[self.case_timestamp_key].last_valid_index()
        if last_valid_idx is None:
            return group
        last_valid_timestamp= group.loc[last_valid_idx, self.case_timestamp_key]
        
        nat_indices = group.index[group[self.case_timestamp_key].isna()]
        for i, idx in enumerate(nat_indices):
            group.at[idx, self.case_timestamp_key] = last_valid_timestamp+ pd.Timedelta(days=i + 1)
    
        return group     
    
    def decode_df(self, df):
        """
        decodes the predictive df; inverse transform timestamps and event names.
        """
        
        df[self.case_activity_key] = df[self.case_activity_key].astype("int")
        df[self.case_id_key] = df[self.case_id_key].astype("int")
        df[self.case_activity_key] = self.config.activity_le.inverse_transform(df[self.case_activity_key])
        df[self.case_id_key] = self.config.case_id_le.inverse_transform(df[self.case_id_key])
        df[self.case_activity_key] = df[self.case_activity_key].astype("str")
        df[self.case_id_key] = df[self.case_id_key].astype("str")
        #: note that this operation is lossy and might generate NaT. 

        df[self.case_timestamp_key] = df[self.case_timestamp_key]*(10**self.config.exponent)
        
        df[self.case_activity_key] = df[self.case_activity_key].astype("str")
        df[self.case_id_key] = df[self.case_id_key].astype("str")

        df[self.case_timestamp_key] = df[self.case_timestamp_key].astype("datetime64[ns, UTC]")

        #: handle NaT values
        df= df.groupby(self.case_id_key, group_keys=False).apply(self.handle_nat)
        #: just in case 
        df = df.dropna() 
        #: save the generated predictive model
        return df

    def import_predictive_df(self, path):
        """
        used for importing a predictive df. 
        """
        self.predictive_df = pd.read_csv(path, sep = ",")

    def visualize(self):
        #: this way it can be accessed from outside the class.
        pm4py.view_petri_net(self.petri_net, self.initial_marking, self.final_marking, format='svg')

    def heuristic_miner(self,path,  dependency_threshold=0.5, and_threshold=0.65, loop_two_threshold=0.5, view= False):
        """
        Run heuristic miner on the predictive log and generate a petri net.

        Args:
            path (str): Path used for saving the generated petri net.
            dependency_threshold (float): Dependency threshold parameter for heuristic miner.
            and_threshold (float): AND threshold parameter for heuristic miner.
            loop_two_threshold (float): Loop two threshold parameter for heuristic miner.
        """
        self.format_columns()
        self.petri_net, self.initial_marking, self.final_marking = pm4py.discover_petri_net_heuristics(
            self.predictive_df,
            dependency_threshold, 
            and_threshold, 
            loop_two_threshold, 
            activity_key=self.case_activity_key,
            timestamp_key=self.case_timestamp_key,
            case_id_key= self.case_id_key
        )
        #: export the petri net in the given path
        pm4py.write_pnml(self.petri_net,self.initial_marking, self.final_marking, file_path=path)
        pm4py.save_vis_petri_net(self.petri_net, self.initial_marking, self.final_marking, file_path = path+".png")

    def format_columns(self): 
        """
        exporting to csv changes the datetime types to object, but we need them to be 
        datetime.  
        """
        self.predictive_df[self.case_timestamp_key] = self.predictive_df[self.case_timestamp_key].astype("datetime64[ns, UTC]")
        self.predictive_df[self.case_id_key] = self.predictive_df[self.case_id_key].astype("str")
        self.predictive_df[self.case_activity_key] = self.config.activity_le.inverse_transform(self.predictive_df[self.case_activity_key].astype(int))
        self.predictive_df[self.case_activity_key] = self.predictive_df[self.case_activity_key].astype("str")

    def inductive_miner(self, path,   noise_threshold=0):
        """
        Run inductive miner on the predictive log and generate a petri net.

        Args:
            path (str): Path used for saving the generated petri net.
            noise_threshold (float): Noise threshold parameter for inductive miner.
        """
        self.format_columns()
        self.petri_net, self.initial_marking, self.final_marking = pm4py.discover_petri_net_inductive(
            self.predictive_df,
            noise_threshold, 
            self.case_activity_key,
            self.case_timestamp_key,
            self.case_id_key
        )
        #pm4py.view_petri_net(self.petri_net, self.initial_marking, self.final_marking, format='svg')
        pm4py.write_pnml(self.petri_net,self.initial_marking, self.final_marking, file_path=path)
        pm4py.save_vis_petri_net(self.petri_net, self.initial_marking, self.final_marking, file_path = path+".png")

    def alpha_miner(self, path):
        """
        Run alpha miner on the predictive log and generate a petri net.

        Args:
            path (str): Path used for saving the generated petri net.
        """
        self.format_columns()
        self.petri_net, self.initial_marking, self.final_marking = pm4py.discover_petri_net_alpha(
            self.predictive_df,
            self.case_activity_key,
            self.case_timestamp_key, 
            self.case_id_key
        )
        #pm4py.view_petri_net(self.petri_net, self.initial_marking, self.final_marking, format='svg')
        pm4py.write_pnml(self.petri_net,self.initial_marking, self.final_marking , file_path=path)
        pm4py.save_vis_petri_net(self.petri_net, self.initial_marking, self.final_marking, file_path = path+".png")

    def prefix_tree_miner(self, path):
        """
        Run prefix tree miner on the predictive log and generate a petri net.

        Args:
            path (str): Path used for saving the generated petri net.
        """
        self.format_columns()
        self.petri_net, self.initial_marking, self.final_marking = pm4py.discover_prefix_tree(
            self.predictive_df,
            self.case_activity_key,
            self.case_timestamp_key,
            self.case_id_key
        )
        #pm4py.view_petri_net(self.petri_net, self.initial_marking, self.final_marking, format='svg')
        pm4py.write_pnml(self.petri_net,self.initial_marking, self.final_marking , file_path=path)
        pm4py.save_vis_petri_net(self.petri_net, self.initial_marking, self.final_marking, file_path = path+".png")
       

    def conformance_checking_token_based_replay(self):
        replayed_traces = pm4py.conformance_diagnostics_token_based_replay(
            self.unencoded_df,  self.petri_net, self.initial_marking, self.final_marking)
        return self.compute_fitness(replayed_traces)
    
    def compute_fitness(self, replayed_traces):
        sum_m =  0
        sum_c = 0
        sum_r = 0
        sum_p  = 0
        # TODO: multiply by trace frequency in the log (there is no such function in pm4py)
        for trace in replayed_traces: 
            sum_m+= 1*trace["missing_tokens"]
            sum_c+= 1*trace["consumed_tokens"]
            sum_r += 1*trace["remaining_tokens"]
            sum_p += 1*trace["produced_tokens"]

        return 0.5*(1-(sum_m/sum_c)) + 0.5*(1-(sum_r/sum_p))
    
    def conformance_checking_alignments(self):
        aligned_traces = pm4py.conformance_diagnostics_alignments(self.unencoded_df, self.petri_net, self.initial_marking, self.final_marking)
        log_fitness = replay_fitness.evaluate(aligned_traces, variant=replay_fitness.Variants.ALIGNMENT_BASED)
        return self.compute_fitness(log_fitness)

    def load_petri_net(self, path): 
        self.petri_net, self.initial_marking, self.final_marking = pm4py.read_pnml(path)
