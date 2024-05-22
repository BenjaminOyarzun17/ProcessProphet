
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
from collections import Counter
import pprint
import datetime as dt
from ERPP_RMTPP_torch import sigmoid



class Preprocessing: 
    """
    This is the preprocessing unit for our server. Provided functionality:
    - adapter for train_split_test: split the event log's data into testing and training data in the right format
    - adapters for importing event logs: make sure the right format is used for the RNN
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
        self.no_classes=0
        self.absolute_frequency_distribution = None
        # TODO: invoke import_event_log? (decide)

    def xes_helper(self, path): 
        log =pm4py.read.read_xes(path)
        dataframe = pm4py.convert_to_dataframe(log)
        print("done loading")
        print(dataframe.columns)

    def import_event_log_xes(self, path, case_id, activity_key, timestamp_key): 
        """
        :param path: path where the xes file is found
        :param case_id: column name for the case id column.
        :param activity_key: column name for the marker column 
        :param timestamp_key: column name for the timestamp column
        effects:  
        - event_df dataframe is generated. The rows are grouped by case id and sorted by time.
        - the generated dataframe has 3 columns two of type string (case id, label) and one of type
        datatime64.
        - event log object: its correctnes is assumed from the pm4py lib and is therefore not tested
        """
        
        self.event_df = pm4py.read.read_xes(path)
        self.event_df = pm4py.convert_to_dataframe(self.event_df)
        self.import_event_log(case_id, activity_key, timestamp_key)
    
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
        self.event_df= pd.read_csv(path, sep=sep)
        self.import_event_log(case_id, activity_key, timestamp_key)

    def import_event_log(self, case_id, activity_key, timestamp_key):
        """
        helper function for import_event_log_csv and import_event_log_xes. 
        - genereates an EventLog object so that other pm4py functions can use it
        - remove all columns other than the three main ones
        - remove all NaN entries
        - format a dataframe using pm4py 
        """
        self.case_id_key =  case_id
        self.case_activity_key = activity_key 
        self.case_timestamp_key = timestamp_key 
        #: returns a formated dataframe that can work with other pm4py functions
        self.event_df = pm4py.format_dataframe(self.event_df, 
                                           case_id=self.case_id_key,
                                             activity_key=self.case_activity_key,
                                             timestamp_key=self.case_timestamp_key) #returns formated df.
        
        #: convert_to_event_log requires string format for case_id and marker
        self.event_df[self.case_id_key] = self.event_df[self.case_id_key].astype("string")
        self.event_df[self.case_activity_key] = self.event_df[self.case_activity_key].astype("string")
        self.event_df[self.case_timestamp_key] = self.event_df[self.case_timestamp_key].astype("datetime64[ns, UTC]")

        self.event_log = pm4py.convert_to_event_log(self.event_df, self.case_id_key) #this returns an event log
        #: filter out all the other generated columns
        self.event_df= self.event_df[[self.case_id_key, self.case_activity_key, self.case_timestamp_key]]
        self.event_df= self.event_df.dropna()


    def string_to_index(self , df, column):
        """
        translate each marker into a specific integer index.  
        """
        col = df[column].tolist()
        uniques = set(col)
        enume = [(label, index) for index, label in enumerate(uniques)]
        return dict(enume)

    


    def split_train_test(self, train_percentage):
        """
        this is an adapter for pm4py's split_train_test so that the data is generated in the right
        format for the model.

        :param train_percentage: what percentage should be used for training
        :returns: two event logs, one for training and one for training (dataframes). the number of classes (for the markers) also returned. the absolute
        frequence distribution for each class in the whole event log. 
        """
        #: we encode the markers with integers to be consistent with the authors implementation
        le1 = LabelEncoder()
        le2 = LabelEncoder()
        self.event_df[self.case_activity_key] = le1.fit_transform(self.event_df[self.case_activity_key])
        self.event_df[self.case_id_key] = le2.fit_transform(self.event_df[self.case_id_key])


        #: get the number of classes
        number_classes = len(self.event_df[self.case_activity_key].unique()) 
        #: trasnform back into strings, its necessary for pm4py
        self.event_df[self.case_activity_key] =self.event_df[self.case_activity_key].astype("str")
        self.event_df[self.case_id_key] =self.event_df[self.case_id_key].astype("str")
        print(f"no_classes: {number_classes}")

        #: compute abs. freq. distribution for the activities. its necessary for CrossEntropyLoss
        absolute_frequency_distribution= Counter(self.event_df[self.case_activity_key].to_list())

        #: do the split
        train, test = pm4py.ml.split_train_test(self.event_df, train_percentage, self.case_id_key)
        if test.shape[0] == 0: 
            raise TrainPercentageTooHigh()


        #: remove the timezone information. we are not using it for simplicity.
        train[self.case_timestamp_key]=train[self.case_timestamp_key].dt.tz_localize(None)
        test[self.case_timestamp_key] = test[self.case_timestamp_key].dt.tz_localize(None)


        #: here we convert the datetime64 (ISO standard) into an integer in POSIX standard. the authors
        # use an Excel format, but we decide to use integers for simplicity.
        train[self.case_timestamp_key]=train[self.case_timestamp_key].astype(int)
        test[self.case_timestamp_key] = test[self.case_timestamp_key].astype(int) 
        #: generates an integer in posix standard. 
        print(test[self.case_timestamp_key].iloc[:30])
        #print(train[self.case_timestamp_key].iloc[:30])
        exponent = test[self.case_timestamp_key].mean()
        train[self.case_timestamp_key]=train[self.case_timestamp_key]/(10**exponent)
        test[self.case_timestamp_key] = test[self.case_timestamp_key]/(10**exponent)

        print(exponent)

        train[self.case_timestamp_key]=train[self.case_timestamp_key].astype("int64")/(10**exponent)
        test[self.case_timestamp_key] = test[self.case_timestamp_key].astype("int64")/(10**exponent)

        #: transform the case id and markers back into float
        train[self.case_activity_key] =train[self.case_activity_key].astype("float64")
        train[self.case_id_key] =train[self.case_id_key].astype("float64")
        test[self.case_activity_key] =test[self.case_activity_key].astype("float64")
        test[self.case_id_key] =test[self.case_id_key].astype("float64")
        return train, test, number_classes, absolute_frequency_distribution




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
        if (len(self.find_start_activities()) != 1) or (len(self.find_end_activities(self)) != 1):
            self.dataframe = pm4py.insert_artificial_start_end(self.event_log, activity_key=self.case_activity_key, case_id_key=self.case_id_key, timestamp_key=self.case_timestamp_key)
            
    def get_sample_case(self):
        """
        returns a sample of a case
        """
        sampled_dataframe = pm4py.sample_cases(self.event_log, 1, case_id_key=self.case_id_key)
        return sampled_dataframe



    def prepare_train_data(self):
        pass
    
    def set_training_parameters(self):
        pass

    def check_path(self): 
        pass



