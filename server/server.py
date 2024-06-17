"""
This module runs the server in the selected port (environment variable).
"""
from server import server_routes 
from flask import Flask
import os
from dotenv import load_dotenv


#: load server port form env variable. 
load_dotenv()
SERVER_PORT= os.getenv('SERVER_PORT')


#: initialize flask app and load the routes from the `server_routes` file
app = Flask(__name__)
app.register_blueprint(server_routes.routes)



if __name__=="__main__": 
    app.run(port = SERVER_PORT,debug=True) #: run the flask server
