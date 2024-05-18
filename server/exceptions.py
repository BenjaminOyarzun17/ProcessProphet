

class TrainPercentageTooHigh(Exception):
    def __init__(self):
        self.message= "the training percentage given is too high. Try a more suitable value. "

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
