from enum import Enum, auto





class TimePrecision(Enum): 
    """
    enumeration for the time precision used in the model.
    """
    NS= auto() #:nanoseconds
    S= auto() #:seconds
    H= auto() #: hours