"""
this module provides a list of exceptions that are caused mainly due to 
improper/invalid parameter selection. 

these exceptions should be raised and handled by the user. 
"""



class NaNException(Exception):
    def __init__(self):
        self.message= "a NaN value was generated during training. try changing the training parameters"

    def __str__(self):
        return self.message


class TrainPercentageTooHigh(Exception):
    def __init__(self):
        self.message= "the training percentage given is too high. Try a more suitable value. "

    def __str__(self):
        return self.message
    
class TrainTimeLimitExceeded(Exception):
    def __init__(self):
        self.message= "the training time limit has been exceeded"

    def __str__(self):
        return self.message


class CutLengthZero(Exception):
    def __init__(self):
        self.message= "cut length zero while attempting tail cut"

    def __str__(self):
        return self.message



class CutTooLarge(Exception):
    def __init__(self):
        self.message= "a cut was done that generated a partial sequence shorter than seq len"

    def __str__(self):
        return self.message



class NotOneCaseId(Exception):
    def __init__(self):
        self.message= "the df use as input does not have one unique case id value."

    def __str__(self):
        return self.message

class SeqLengthTooHigh(Exception):
    def __init__(self):
        self.message= "the given sequence length is too high."

    def __str__(self):
        return self.message


class ProcessTooShort(Exception):
    def __init__(self):
        self.message= "the  input process is too short "

    def __str__(self):
        return self.message

class ModelNotTrainedYet(Exception):
    def __init__(self):
        self.message= "the model has not been trained yet!"

    def __str__(self):
        return self.message


class TransitionNotDefined(Exception):
    def __init__(self):
        self.message= "this transition has not been defined"

    def __str__(self):
        return self.message
