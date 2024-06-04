import pytermgui as ptg
from ProcessProphet import ProcessProphet
from ProcessProphetPreprocessing import ProcessProphetPreprocessing
from ProcessProphetTrain import ProcessProphetTrain
import os
from loggers import logger_set_params_cli
from dotenv import load_dotenv







load_dotenv()
SERVER_NAME= os.getenv('SERVER_NAME')


class ProcessProphetStart: 
    def __init__(self, pp):
        self.pp = pp
        self.pp.switch_window(self.main_menu())

    def launch_preprocessor(self):
        preprocessor= ProcessProphetPreprocessing(self.pp)
        
    def launch_trainer(self):
        trainer = ProcessProphetTrain(self.pp)


    def launch_predictor(self):
        pass
    def launch_conformance(self):
        pass


    def select_manager(self) : 
        """
        TODO: the manager alternativs should depend on the sate of the folders, 
        ie if for example the models folder is empty, no conformance checking should be possible. 
        other ex: if decoded_dfs is empty, train should not be possible.
        """
        container = [
            ptg.Label(f"select one of the following actions:"),
            "",
            ptg.Button(f"{self.pp.button_color}import and filter log", lambda *_: self.launch_preprocessor()), 
            "",
            ptg.Button(f"{self.pp.button_color}train neural network", lambda *_: self.launch_trainer()), 
            "",
            ptg.Button(f"{self.pp.button_color}make predictions", lambda *_: self.pp.switch_window(self.launch_predictor())), 
            "",
            ptg.Button(f"{self.pp.button_color}conformance checking", lambda *_: self.pp.switch_window(self.launch_conformance())), 
            "",
            ptg.Button(f"{self.pp.button_color}back to menu", lambda *_: self.pp.switch_window(self.main_menu())), 
            "",
            ptg.Button(f"{self.pp.button_color}Exit", lambda *_: self.pp.manager.stop())
        ] 

        window = ptg.Window(*container, box="DOUBLE")
        window.center()
        return window

    def notify_project_creation(self, message, success): 
        if success: 
            container =[  
                ptg.Label(f"{message}"),
                "",
                ptg.Button(f"{self.pp.button_color}continue", lambda *_: self.pp.switch_window(self.select_manager())), 
                ptg.Button(f"{self.pp.button_color}Exit", lambda *_: self.pp.manager.stop())
            ]
        else: 
            container = [ 
                ptg.Label(f"{message}!"),
                ptg.Button(f"{self.pp.button_color}back to menu", lambda *_: self.pp.switch_window(self.main_menu()))
            ]


        window = ptg.Window(*container, box="DOUBLE")
        window.center()
        return window

    def handle_project_name_input(self):
        name = self.project_name_input.value
        message = ""
        success = True
        if name in os.listdir(f"{os.getcwd()}/{self.pp.state.projects_path}"):
            message = "directory already exits"
            success = False
        else: 
            message = f"directory created in path {os.getcwd()}/{self.pp.state.projects_path}/{name}"
            subdirectories = ["input_logs", "models", "petri_nets", "predictive_logs", "partial_traces", "decoded_dfs"]
            os.mkdir(f"{os.getcwd()}/{self.pp.state.projects_path}/{name}")
            self.pp.state.current_project = name
            self.pp.state.input_logs_path  = f"{os.getcwd()}/{self.pp.state.projects_path}/{name}/input_logs"
            self.pp.state.models_path  = f"{os.getcwd()}/{self.pp.state.projects_path}/{name}/models"
            self.pp.state.petri_nets_path  = f"{os.getcwd()}/{self.pp.state.projects_path}/{name}/petri_nets"
            self.pp.state.predictive_logs_path  = f"{os.getcwd()}/{self.pp.state.projects_path}/{name}/predictive_logs"
            self.pp.state.partial_traces_path = f"{os.getcwd()}/{self.pp.state.projects_path}/{name}/partial_traces"
            self.pp.state.decoded_dfs_path = f"{os.getcwd()}/{self.pp.state.projects_path}/{name}/decoded_dfs"
            for subdirectory in subdirectories: 
                os.mkdir(f"{os.getcwd()}/{self.pp.state.projects_path}/{name}/{subdirectory}")

        
        self.pp.switch_window(self.notify_project_creation(message, success))




    def handle_project_selection(self):

        projects = [project  for project in os.listdir(f"{self.pp.state.projects_path}")]
        name= self.input_select_project.value
        if name in projects: 

            self.pp.state.input_logs_path  = f"{os.getcwd()}/{self.pp.state.projects_path}/{name}/input_logs"
            self.pp.state.models_path  = f"{os.getcwd()}/{self.pp.state.projects_path}/{name}/models"
            self.pp.state.petri_nets_path  = f"{os.getcwd()}/{self.pp.state.projects_path}/{name}/petri_nets"
            self.pp.state.predictive_logs_path  = f"{os.getcwd()}/{self.pp.state.projects_path}/{name}/predictive_logs"
            self.pp.state.partial_traces_path = f"{os.getcwd()}/{self.pp.state.projects_path}/{name}/partial_traces"
            self.pp.state.decoded_dfs_path = f"{os.getcwd()}/{self.pp.state.projects_path}/{name}/decoded_dfs"
            container =[  
                "Project selected successfully", 
                "",
                ptg.Button(f"{self.pp.button_color}continue", lambda *_: self.pp.switch_window(self.select_manager())), 
                "",
                ptg.Button(f"{self.pp.button_color}Exit", lambda *_: self.pp.manager.stop())
            ]
            window = ptg.Window(*container, box="DOUBLE")
            window.center()
            return window
        else: 
            container =  [ 
                "Project does not exist", 
                ptg.Button(f"{self.pp.button_color}Exit", lambda *_: self.pp.manager.stop())
            ]
            window = ptg.Window(*container, box="DOUBLE")
            window.center()
            return window


    def load_existing_project(self): 

        projects = [f"{project}"  for project in os.listdir(f"{self.pp.state.projects_path}")]

    
        self.input_select_project= ptg.InputField(projects[0],  prompt="enter a project name: ")
       
        left_container = ptg.Container(
            "[underline]Select a project", 
            "",
            self.input_select_project, 
            ptg.Button(f"{self.pp.button_color}Select", lambda *_: self.pp.switch_window(self.handle_project_selection())), 
            "", 
            ptg.Button(f"{self.pp.button_color}Exit", lambda *_: self.pp.manager.stop())
        )

        right_container= ptg.Container(
            "[underline]Existing projects", 
            *projects 
        )
        window = ptg.Window(ptg.Splitter(left_container,right_container), width = 80)
        #window = ptg.Window(*c, box="DOUBLE")
        window.center()
        return window

    def new_project_form(self):

        self.project_name_input =  ptg.InputField("first Prophet", prompt="Project name: ")
        container =[ 
            ptg.Label(f"{self.pp.button_color}Create new project"),
            ptg.Label(f"{self.pp.button_color}current path: {os.getcwd()}/{self.pp.state.projects_path}"),
            "", 
            self.project_name_input, 
            "", 
            ptg.Button(f"{self.pp.button_color}Create project", lambda *_: self.handle_project_name_input()), 
            "", 
            ptg.Button(f"{self.pp.button_color}Back to start", lambda *_: self.pp.switch_window(self.main_menu())), 
            "", 
            ptg.Button(f"{self.pp.button_color}Exit", lambda *_: self.pp.manager.stop())
        ] 

        window = ptg.Window(*container, box="DOUBLE")
        window.center()
        return window


    def main_menu(self):
        container =[ 
            ptg.Label("Welcome to [yellow]Process Prophet"),
            "", 
            ptg.Label("Choose one option:"),
            "", 
            ptg.Button(f"{self.pp.button_color}Create new project", lambda *_: self.pp.switch_window(self.new_project_form())),
            "", 
            ptg.Button(f"{self.pp.button_color}Load existing project", lambda *_: self.pp.switch_window(self.load_existing_project())),
            "", 
            ptg.Button(f"{self.pp.button_color}Exit", lambda *_: self.pp.manager.stop())
        ] 
        
        window = ptg.Window(*container, title = "Process Prophet")
        window.center()
        return window

    def run(self):
        super().run()