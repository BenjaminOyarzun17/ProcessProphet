
import pm4py
import prediction_manager


class ProcessModelManager:

    def __init__(self):
        self.predictive_df= None
        self.model = None 
        self.event_df = None

        self.case_activity_key =None 
        self.case_id_key =None 
        self.case_timestamp_key=None 
    def generate_predictive_log(self): 
        """
        1. get all unique case ids
            -> preprocessor?
            -> nn_manager?
            -> pred manager?
        2. for each case id get the length of the sequence
        3. for each case id do a random cut 
        and remember the number of desired predictions
            3.1 add the cut prefix to a df
        4. do the predictions (path)  and append
        them to the log
            -> composition with pred manager.
        5. export the csv or save in an event log (pm4py). 
        """
        pass

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