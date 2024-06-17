"""
this module starts the CLI
"""
from ProcessProphetStart import ProcessProphetStart
from ProcessProphet import ProcessProphet
from dotenv import load_dotenv








if __name__=="__main__":
    #: run the app
    pp = ProcessProphet()
    pps = ProcessProphetStart(pp) #: always start with the "initial menu"
    pp.run()