import requests
import json






if __name__=="__main__":

    params = {
        "is_xes": "False", 
        "path_to_log": "asdas", 
        "sep": ",", 
        "case_id": "case_id", 
        "activity_key": "activity", 
        "timestamp_key": "timestamp"
    }
    response = requests.get(
        "http://localhost:5000/train_nn", 
        params = params,
        timeout =60
    )
    print(response.status_code)
    print(response.json())