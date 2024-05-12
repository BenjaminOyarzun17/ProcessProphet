
import dateutil.parser
import pm4py
import os
import pandas as pd
import pydoc_markdown
import numpy as np
import dateutil
from sklearn.preprocessing import LabelEncoder



class Preprocessing: 
    """
    This is the preprocessing unit for our server. Provided functionality:
    - adapter for train_split_test: split the event log's data into testing and training data in the right format
    - adapters for importing event logs: make sure the right format is used
    - TODO: might be extended
    """
    def __init__(self):
        #: contains the event log path
        self.event_log_path= None
        self.event_log= None
        self.case_id_key = None 
        self.case_activity_key = None 
        self.case_timestamp_key = None 
        # TODO: invoke import_event_log? (decide)

    def import_event_log_xes(self, path): 
        """
        Import the event log as xes
        the case id, activity key and timestamp keys are well documented in XES
        """
        self.event_log = pm4py.read.read_xes(path)
        """
        TODO: connect this with train_split_... 
        """




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
        dataframe= pd.read_csv(path, sep=sep)
        # this line transforms the event log in the required input 
        # for the RNN: groups the data by id and sorts the entries
        # accorting to time. 
        dataframe = dataframe.groupby(case_id).apply(lambda x: x.sort_values(timestamp_key)).reset_index(drop=True)
        # we save the df sorted, because its way more useful than the unsorted version.
        self.event_df = dataframe.copy()
        self.event_df[timestamp_key] = pd.to_datetime(self.event_df[timestamp_key])

        dataframe = pm4py.format_dataframe(dataframe, 
                                           case_id=case_id,
                                             activity_key=activity_key,
                                               timestamp_key=timestamp_key)
        self.case_id_key =  case_id
        self.case_activity_key = activity_key 
        self.case_timestamp_key = timestamp_key 
        self.event_log = pm4py.convert_to_event_log(dataframe)


    def split_train_test(self, train_percentage):
        """
        this is an adapter for pm4py's split_train_test so that the data is generated in the right
        format for the model.

        :param train_percentage: what percentage should be used for training
        :returns: two event logs, one for training and one for training (dataframes). the number of classes (for the markers) also returned.
        """
        number_classes = len(self.event_df[self.case_activity_key].unique())
        le = LabelEncoder() 
        train, test = pm4py.ml.split_train_test(self.event_df, train_percentage, self.case_id_key)
        #: the author uses floats for representing time
        train[self.case_timestamp_key] = train[self.case_timestamp_key].map(lambda x: x.timestamp())
        test[self.case_timestamp_key] = test[self.case_timestamp_key].map(lambda x: x.timestamp())

        #: we encode the markers with integers to be consistent with the authors implementation
        train[self.case_activity_key] =  le.fit_transform(train[self.case_activity_key])
        test[self.case_activity_key] =  le.fit_transform(test[self.case_activity_key])
        return train, test, number_classes



    def convert_csv_to_pandas(self):
        pass
    def prepare_train_data(self):
        pass
    def split_event_log(self):
        pass
    def set_training_parameters(self):
        pass