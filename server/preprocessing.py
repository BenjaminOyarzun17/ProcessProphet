
#import dateutil.parser
#import dateutil
import pm4py
import os
import pandas as pd
import pydoc_markdown
import numpy as np
from sklearn.preprocessing import LabelEncoder
from exceptions import *
from datetime import datetime, timezone 



class Preprocessing: 
    """
    This is the preprocessing unit for our server. Provided functionality:
    - adapter for train_split_test: split the event log's data into testing and training data in the right format
    - adapters for importing event logs: make sure the right format is used
    - TODO: might be extended
    """
    def __init__(self):
        #: contains the event log path
        self.event_log_path = None
        self.event_log= None
        self.case_id_key = None 
        self.case_activity_key = None 
        self.case_timestamp_key = None 
        self.event_df = None
        self.is_xes= False
        # TODO: invoke import_event_log? (decide)

    def import_event_log_xes(self, path): 
        """
        Import the event log as xes
        the case id, activity key and timestamp keys are well documented in XES
        """
        self.is_xes = True
        self.event_log = pm4py.read.read_xes(path)
        dataframe = pm4py.convert_to_dataframe(self.event_log)
        dataframe["time:timestamp"]= dataframe["time:timestamp"].map(lambda x: x.timestamp())
        self.event_df = dataframe
        print(dataframe.columns)
        print(dataframe.head(20))
        #dataframe.to_csv("../data/dummy.csv",',',columns= ["concept:name", "time:timestamp", "Activity code"], header = True, index_label = ["concept:name", "time:timestamp", "Activity code"] , index = False)
        #dataframe.to_csv("../data/info.csv",',',columns= dataframe.columns, header = True, index_label = ["concept:name", "time:timestamp", "Activity code"] , index = False)
        dataframe.to_csv("../data/dummy.csv",',',columns= ["concept:name", "time:timestamp", "case:concept:name"], header = True, index_label = ["concept:name", "time:timestamp", "case:concept:name"] , index = False)
        self.import_event_log_csv("../data/dummy.csv", "case:concept:name", "concept:name", "time:timestamp", ",")
        


    def import_event_log_csv(self, path, case_id, activity_key, timestamp_key, sep): 
        """
        this is an adapter for format_dataframe such that 
        the event data can be properly used by the rnn model. 

        :param path: path to the event log
        :param case_id: case id column name
        :param activity_key: activity column name
        :param timestamp_key: timestamp column name
        :param sep: separator
        """

        self.case_id_key =  case_id
        self.case_activity_key = activity_key 
        self.case_timestamp_key = timestamp_key 
        dataframe= pd.read_csv(path, sep=sep)
        # this line transforms the event log in the required input 
        # for the RNN: groups the data by id and sorts the entries
        # accorting to time. 
        dataframe =  dataframe.sort_values(by=[case_id, timestamp_key])
        #dataframe = dataframe.groupby(case_id).apply(lambda x: x.sort_values(timestamp_key)).reset_index(drop=True)
        # we save the df sorted, because its way more useful than the unsorted version.
        self.event_df = dataframe.copy()
        self.event_df[timestamp_key] = pd.to_datetime(self.event_df[timestamp_key])
        self.event_df = pm4py.format_dataframe(self.event_df, 
                                           case_id=case_id,
                                             activity_key=activity_key,
                                             timestamp_key=timestamp_key) #returns formated df.

        #self.event_df.to_csv("after_sort.csv")
        self.event_log = pm4py.convert_to_event_log(self.event_df) #this returns an event log



    


    def split_train_test(self, train_percentage):
        """
        this is an adapter for pm4py's split_train_test so that the data is generated in the right
        format for the model.

        :param train_percentage: what percentage should be used for training
        :returns: two event logs, one for training and one for training (dataframes). the number of classes (for the markers) also returned.
        """
        #TODO: check the correcctness of this function
        #: we encode the markers with integers to be consistent with the authors implementation
        le = LabelEncoder() 
        self.event_df[self.case_activity_key] =  le.fit_transform(self.event_df[self.case_activity_key])
        number_classes = len(le.classes_)
        le2 = LabelEncoder() 
        self.event_df[self.case_id_key] =  le2.fit_transform(self.event_df[self.case_id_key])
        #number_classes = len(self.event_df[self.case_activity_key].unique())
        #train[self.case_activity_key] =  le.fit_transform(train[self.case_activity_key])
        #test[self.case_activity_key] =  le.fit_transform(test[self.case_activity_key])
        print(f"no_classes: {number_classes}")
        #print(self.event_df.columns)
        #print(self.event_df[self.case_timestamp_key])
        print("before split")
        

        #self.event_df[self.case_activity_key]  = self.event_df[self.case_activity_key].astype(str)
        #self.event_df[self.case_id_key]  =  self.event_df[self.case_id_key].astype(str)



        train, test = pm4py.ml.split_train_test(self.event_df, train_percentage, self.case_id_key)
        if test.shape[0] == 0: 
            raise TrainPercentageTooHigh()
        
        #train.reset_index(drop =True, inplace= True)
        #test.reset_index(drop =True, inplace= True)

        print(train.head())
        #train[self.case_timestamp_key].to_csv("before_transform.txt")


        #: here we convert the datetime64 into an integer. the authors
        # use an Excel format, but we decide to use integers for simplicity.
        train[self.case_timestamp_key]=train[self.case_timestamp_key].astype('int64')
        test[self.case_timestamp_key] = test[self.case_timestamp_key].astype('int64')

        if self.is_xes: 
            try:
                train = train.drop(["concept:name.1", "time:timestamp.1", "Activity code.1"],axis = 1 )
                test= test.drop(["concept:name.1", "time:timestamp.1", "Activity code.1"],axis = 1 )
            except:
                pass


        train[self.case_timestamp_key].to_csv("after_transform.txt")
        print(train[self.case_timestamp_key].dtype)
        print(train[self.case_timestamp_key])
        

        return train, test, number_classes

    def find_start_activities(self):
        """
        find the start activities of all cases for an existing log and return a dict with start activities as keys and value is the count of this activity
        """
        start_activities = pm4py.stats.get_start_activities(self.event_log, activity_key=self.case_activity_key, case_id_key=self.case_id_key, timestamp_key=self.case_timestamp_key)
        return start_activities
        
    def find_end_activities(self):
        """"
        find the end activities of all cases for an existing log and return a dict with end activities as keys and value is the count of this activity
        """
        end_activities = pm4py.get_end_activities(self.event_log, activity_key=self.case_activity_key, case_id_key=self.case_id_key, timestamp_key=self.case_timestamp_key)
        return end_activities
        
    def add_unique_start_end_activity(self):
        """
        if there is no unique start/ end activity, add an artificial start and end activity
        """
        if (len(find_start_activities()) != 1) or (len(find_end_activities(self)) != 1):
            self.dataframe = pm4py.insert_artificial_start_end(self.event_log, activity_key=self.case_activity_key, case_id_key=self.case_id_key, timestamp_key=self.case_timestamp_key)
            
    def get_sample_case(self):
        """
        returns a sample of a case
        """
        sampled_dataframe = pm4py.sample_cases(self.event_log, 1, case_id_key=self.case_id_key)
        return sampled_dataframe



    def convert_csv_to_pandas(self):
        pass
    def prepare_train_data(self):
        pass
    def split_event_log(self):
        pass
    def set_training_parameters(self):
        pass

    def check_path(self): 
        pass