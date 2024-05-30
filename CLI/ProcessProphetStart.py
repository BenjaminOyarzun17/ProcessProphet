import pytermgui as ptg
from ProcessProphet import ProcessProphet
from ProcessProphetPreprocessing import ProcessProphetPreprocessing
from ProcessProphetTrain import ProcessProphetTrain
import os
from loggers import logger_set_params_cli


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
        container = ptg.Container(
            ptg.Label(f"select one of the following actions:"),
            "",
            ptg.Button("import and filter log", lambda *_: self.launch_preprocessor()), 
            "",
            ptg.Button("train neural network", lambda *_: self.launch_trainer()), 
            "",
            ptg.Button("make predictions", lambda *_: self.pp.switch_window(self.launch_predictor())), 
            "",
            ptg.Button("conformance checking", lambda *_: self.pp.switch_window(self.launch_conformance())), 
            "",
            ptg.Button("back to menu", lambda *_: self.pp.switch_window(self.main_menu())), 
            "",
            ptg.Button("Exit", lambda *_: self.manager.stop())
        ).center()

        window = ptg.Window(container, box="DOUBLE")
        window.center()
        return window

    def notify_project_creation(self, message, success): 
        if success: 
            container = ptg.Container(
                ptg.Label(f"{message}"),
                "",
                ptg.Button("continue", lambda *_: self.pp.switch_window(self.select_manager())), 
                ptg.Button("Exit", lambda *_: self.manager.stop())
            ).center()
        else: 
            container = ptg.Container(
                ptg.Label(f"{message}!"),
                ptg.Button("back to menu", lambda *_: self.pp.switch_window(self.main_menu()))
            ).center()


        window = ptg.Window(container, box="DOUBLE")
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
            container = ptg.Container(
                "Project selected successfully", 
                ptg.Button("continue", lambda *_: self.pp.switch_window(self.select_manager())), 
                ptg.Button("Exit", lambda *_: self.pp.manager.stop())
            ).center()
            window = ptg.Window(container, box="DOUBLE")
            window.center()
            return window
        else: 
            container = ptg.Container(
                "Project does not exist", 
                ptg.Button("Exit", lambda *_: self.pp.manager.stop())
            ).center()
            window = ptg.Window(container, box="DOUBLE")
            window.center()
            return window


    def load_existing_project(self, mask = []): 

        c = [ptg.Label("Pick one of the following: ")]
        projects = [project  for project in os.listdir(f"{self.pp.state.projects_path}")]
        if len(mask) ==0:
            mask = [False]*len(projects)
            mask[0]= True

        #checkboxes = [ptg.Label(project)  for project in os.listdir(f"{self.pp.state.projects_path}")]
        checkboxes = []

        for id, project in enumerate(os.listdir(f"{self.pp.state.projects_path}")): 
            checkboxes.append(
                ptg.Label(
                    f"- {project}"
                )
            )
        c= c+ checkboxes
        self.input_select_project= ptg.InputField(projects[0],  prompt="enter a project name: ")
        c.append("")
        c.append(self.input_select_project)
        c.append("")
        c.append(ptg.Button("select", lambda *_: self.pp.switch_window(self.handle_project_selection())))
        c.append("")
        c.append(ptg.Button("Exit", lambda *_: self.pp.manager.stop()))
        container = ptg.Container(
            *c    
        ).center()

        window = ptg.Window(container, box="DOUBLE")
        window.center()
        return window

    def new_project_form(self):

        self.project_name_input =  ptg.InputField("first Prophet", prompt="Project name: ")
        container = ptg.Container(
            ptg.Label("Create new project"),
            ptg.Label(f"current path: {os.getcwd()}/{self.pp.state.projects_path}"),
            "", 
            self.project_name_input, 
            "", 
            ptg.Button("Create project", lambda *_: self.handle_project_name_input()), 
            "", 
            ptg.Button("Back to start", lambda *_: self.pp.switch_window(self.main_menu())), 
            "", 
            ptg.Button("Exit", lambda *_: self.manager.stop())
        ).center()

        window = ptg.Window(container, box="DOUBLE")
        window.center()
        return window


    def main_menu(self):
        container = ptg.Container(
            ptg.Label("Welcome to Process Prophet"),
            "", 
            ptg.Label("Choose one option:"),
            "", 
            ptg.Button("Create new project", lambda *_: self.pp.switch_window(self.new_project_form())),
            "", 
            ptg.Button("Load existing project", lambda *_: self.pp.switch_window(self.load_existing_project())),
            "", 
            ptg.Button("Exit", lambda *_: self.manager.stop())
        ).center()
        
        window = ptg.Window(container, box="DOUBLE")
        window.center()
        return window

    def run(self):
        super().run()