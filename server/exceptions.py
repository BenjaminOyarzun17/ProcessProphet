

class TrainPercentageTooHigh(Exception):
    def __init__(self):
        self.message= "the training percentage given is too high. Try a more suitable value. "

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
