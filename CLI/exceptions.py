

class BooleanExpected(Exception):
    def __init__(self):
        self.message= "a boolean was expected (True/False)"

    def __str__(self):
        return self.message



class IntegerExpected(Exception):
    def __init__(self):
        self.message= "an integer was expected"

    def __str__(self):
        return self.message


class XESorCSVExpected(Exception):
    def __init__(self):
        self.message= "csv/xes expected. check extension?"

    def __str__(self):
        return self.message


class FileNotFound(Exception):
    def __init__(self, folder):
        self.message= "file not found in folder {folder}"

    def __str__(self):
        return self.message


class AlreadyExists(Exception):
    def __init__(self, folder):
        self.message= "the file/folder already exists"

    def __str__(self):
        return self.message