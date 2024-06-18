"""
this module is in charge of: 
    - supporting event log imports from xes/csv files. 
    - formatting the event log so that it can be later on used by the `nn_manager` module. 
    in particular, the timestamps are encoded as integers, the case id's and activity names
    are encoded, and the rows are sorted by case id and timestamp. Splitting the event log in 
    training and testing sublogs is also supported. 
    - the preprocessor also calculates important values such as the number of activities and 
    absolute frequency distribution, which are also required by the neural network's training. 
    - formatting is done automatically after importing, but this can also be deselected by 
    setting the corresponding parameter. 
    - other preprocessing operations are supported, such as replacing NaN values, adding a unique
    start / end activity to the log, and removing duplicate rows. 

Note that this module does not bring the event log in the input format
for the RNN. this is done by the module `util.py` in the subpackage
`RMTPP_torch`.
"""
from server import exceptions
from server import loggers
from server import time_precision

import random
import pandas as pd
import datetime as dt
from sklearn.preprocessing import LabelEncoder
from collections import Counter
import pm4py

import datetime as dt







class Preprocessing: 
    """
    This is the preprocessing unit for our server. Provided functionality:
    - adapter for train_split_test: split the event log's data into testing and training data in the right format
    - adapters for importing event logs: make sure the right format is used for the RNN
    """
    def __init__(self):
        self.time_precision = None

        #: contains the event log path
        self.event_log_path = None
        self.event_log= None
        self.case_id_key = None 
        self.case_activity_key = None 
        self.case_timestamp_key = None 

        self.event_df = None #: dataframe containing the event log. corresponds to the imported log and eventually also the formatted one
        self.number_classes=0
        self.absolute_frequency_distribution = None
        self.case_id_le = None
        self.activity_le = None
        self.exponent = None
        self.unencoded_df = None
    


    def xes_helper(self, path): 
        """just a testing function"""
        log =pm4py.read.read_xes(path)
        dataframe = pm4py.convert_to_dataframe(log)
        print("done loading")
        print(dataframe.columns)


    def handle_import(self,is_xes, path, case_id, timestamp, activity,time_precision = time_precision.TimePrecision.NS,  sep = ",", formatting = True):
        self.time_precision = time_precision

        self.case_id_key =  case_id
        self.case_activity_key =activity 
        self.case_timestamp_key =timestamp 
        if is_xes: 
            self.import_event_log_xes(path, formatting)
        else: 
            self.import_event_log_csv(path, sep, formatting)


    def import_event_log_xes(self, path, formatting=True):
        """
        Imports an event log in XES format.

        Args:
        path (str): Path to the XES file.
        formatting (bool, optional): If True, the event log is formatted so that it can be used by the RNN. Defaults to True.

        Effects:
        - event_df dataframe is generated.
        - The generated dataframe has 3 columns: case id (string), label (string), and timestamp (datetime64).
        - event log object: its correctness is assumed from the pm4py library and is therefore not tested.
        """
        self.event_df = pm4py.read.read_xes(path)
        self.event_df = pm4py.convert_to_dataframe(self.event_df)
        self.import_event_log(formatting)


    def import_event_log_csv(self, path, sep, formatting = True): 
        """
        This is an adapter for format_dataframe such that the event data can be properly used by the RNN.

        Args:
            path (str): Path to the event log.
            sep (str): Separator.
            formatting (bool, optional): If True, the event log is formatted so that it can be used by the RNN. Defaults to True.
        """
        self.event_df= pd.read_csv(path, sep=sep)
        self.import_event_log(formatting)


    def import_event_log_dataframe(self,df, case_id, activity_key, timestamp_key, formatting = True):
        """
        This is an adapter for format_dataframe such that the event data can be properly used by the RNN model.

        Args:
            path (str): Path to the event log.
            case_id (str): Case id column name.
            activity_key (str): Activity column name.
            timestamp_key (str): Timestamp column name.
        """
        self.event_df = df
        self.case_id_key =  case_id
        self.case_activity_key =activity_key
        self.case_timestamp_key =timestamp_key
        self.import_event_log(formatting)


    def import_event_log(self, formatting):
        """
        helper function for import_event_log_csv and import_event_log_xes. 
        - genereates an EventLog object so that other pm4py functions can use it
        - remove all columns other than the three main ones
        - remove all NaN entries
        - format a dataframe using pm4py 
        Effects: 
        - rows sorted by case id and timestamp
        """
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

        
        #: the rest should only be executed when training
        if not formatting:
            return


        self.event_df= self.event_df.dropna()
        #: used for conformance checking, save everything except the 
        # extra columns 
        self.unencoded_df = self.event_df.copy(deep = True)

        
        #: sort the rows by group id and timestamp key
        self.event_df =  self.event_df.sort_values(by=[self.case_id_key, self.case_timestamp_key])
        


        #loggers.logger_import_event_log.debug(self.event_df.iloc[:30])

        self.encode_df_columns()



    def string_to_index(self , df, column):
        """
        translate each marker into a specific integer index.  
        """
        col = df[column].tolist()
        uniques = set(col)
        enume = [(label, index) for index, label in enumerate(uniques)]
        return dict(enume)

    
    def encode_df_columns(self):
        """
        - encode the markers and case id's with integers (label encoding)
        - encode the timestamps
        - returns nothing, but modifies self.event_df
        """
        #: we encode the markers with integers (label encoding) to be consistent with the authors implementation
        self.activity_le, self.case_id_le= LabelEncoder(), LabelEncoder()
        self.event_df[self.case_activity_key] = self.activity_le.fit_transform(self.event_df[self.case_activity_key])
        self.event_df[self.case_id_key] = self.case_id_le.fit_transform(self.event_df[self.case_id_key])

        #: get the number of classes
        self.number_classes = len(self.event_df[self.case_activity_key].unique()) 
        #: trasnform back into strings, its necessary for pm4py
        self.event_df[self.case_activity_key] =self.event_df[self.case_activity_key].astype("str")
        self.event_df[self.case_id_key] =self.event_df[self.case_id_key].astype("str")

        #: compute abs. freq. distribution for the activities. its necessary for CrossEntropyLoss
        self.absolute_frequency_distribution= Counter(self.event_df[self.case_activity_key].to_list())

        # remove timezone information
        self.event_df[self.case_timestamp_key] = self.event_df[self.case_timestamp_key].dt.tz_localize(None)

        #: here we convert the datetime64 (ISO standard) into an integer in POSIX standard. the authors
        # use an Excel format, but we decide to use integers for simplicity.
        self.event_df[self.case_timestamp_key] = self.event_df[self.case_timestamp_key].astype(int)


        if self.time_precision == time_precision.TimePrecision.NS: 
            #: nanoseconds can cause numerical instability. therefore we make the number smaller by shifting the comma.
            self.exponent = self.event_df[self.case_timestamp_key].astype(str).apply(lambda x: len(x)).mean()
            self.event_df[self.case_timestamp_key] = self.event_df[self.case_timestamp_key] / (10 ** self.exponent)


        # #: transform the case id and markers back into float
        self.event_df[self.case_activity_key] = self.event_df[self.case_activity_key].astype("float64")
        self.event_df[self.case_id_key] = self.event_df[self.case_id_key].astype("float64")

    def split_train_test(self, train_percentage):
        """
        This is a helper function for splitting the event log into training and testing data.

        Args:
            train_percentage (float): The percentage of data to be used for training.

        Returns:
            tuple: A tuple containing two event logs (dataframes) for training and testing, the number of classes (for the markers), and the absolute frequency distribution for each class in the whole event log.
        """
        if train_percentage>=1 or train_percentage<=0: 
            raise exceptions.TrainPercentageTooHigh()

        cases = self.event_df[self.case_id_key].unique().tolist()
        train_cases = set()
        test_cases = set()
        for c in cases:
            r = random.random()
            if r <= train_percentage:
                train_cases.add(c)
            else:
                test_cases.add(c)
        train = self.event_df[self.event_df[self.case_id_key].isin(train_cases)]
        test = self.event_df[self.event_df[self.case_id_key].isin(test_cases)]

        if test.shape[0] == 0: 
            raise exceptions.TrainPercentageTooHigh()


        return train, test



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
        activities = self.event_df[self.case_activity_key].unique()
        end_activity_lookup = {activity:False  for activity in activities}
        for activity in activities:
            end_activity_lookup[activity]= True
        return end_activity_lookup 
        
            
    def get_sample_case(self):
        """
        returns a sample of a case
        """
        sampled_dataframe = pm4py.sample_cases(self.event_log, 1, case_id_key=self.case_id_key)
        return sampled_dataframe



    def replace_activity_nan_with_mode(self): 
        """
        replaces NaN values in activity column with median
        """
        
        mode=  self.event_df[self.case_activity_key].mode()
        self.event_df[self.case_activity_key].fillna(mode, inplace = True)

        return True


    def remove_duplicate_rows(self): 
        #: removes the duplicates ie the rows where the same activity happened at the same time in the same case id.
        # since we are dropping all the other columns, these duplicates make no sense.
        self.event_df = self.event_df.drop_duplicates(subset=[self.case_id_key, self.case_activity_key, self.case_timestamp_key])
        return True

    
    def add_unique_start_end_activity(self):
        """
        if there is no unique start/ end activity, add an artificial start and end activity
        """
        if (len(self.find_start_activities()) != 1) or (len(self.find_end_activities()) != 1):
            processed_log= pm4py.insert_artificial_start_end(
                self.event_log, 
                activity_key=self.case_activity_key, 
                case_id_key=self.case_id_key, 
                timestamp_key=self.case_timestamp_key
            )
            self.event_df =pm4py.convert_to_dataframe(processed_log) 
            return True
        return False