import pytermgui as ptg
from ProcessProphet import ProcessProphet
from ProcessProphetPreprocessing import ProcessProphetPreprocessing
from ProcessProphetTrain import ProcessProphetTrain
from ProcessProphetPredict import ProcessProphetPredict
from ProcessProphetModel import ProcessProphetModel
import os
from loggers import logger_set_params_cli
from dotenv import load_dotenv
from process_prophet_modes import ProcessProphetMode




load_dotenv()
SERVER_NAME= os.getenv('SERVER_NAME')
SERVER_PORT= os.getenv('SERVER_PORT')




class ProcessProphetStart: 
    def __init__(self, pp, start:bool= True):
        """
        initialize ProcessProphet Object and main menu
        """
        self.pp = pp
        if start: 
            self.pp.switch_window(self.main_menu())
        else: 
            self.pp.switch_window(self.select_manager())
        

    def launch_preprocessor(self):
        """
        launches the Preprocessing CLI interface 
        """
        preprocessor= ProcessProphetPreprocessing(self.pp)
        
    def launch_trainer(self):
        """
        launches the Training CLI interface 
        """
        trainer = ProcessProphetTrain(self.pp)

    def launch_predictor(self):
        """
        launches the Predictor CLI interface 
        """
        predictor = ProcessProphetPredict(self.pp)
    def launch_conformance(self):
        """
        launches the Conformance checking CLI interface 
        """
        conformance_checker = ProcessProphetModel(self.pp)


    def select_manager(self) : 
        """
        TODO: the manager alternativs should depend on the sate of the folders, 
        ie if for example the models folder is empty, no conformance checking should be possible. 
        other ex: if decoded_dfs is empty, train should not be possible.

        creates menu with the main actions for a given project
        """
        container = [
            ptg.Label(f"select one of the following actions:"),
            "",
            ptg.Button(f"{self.pp.button_color}import and filter log", lambda *_: self.launch_preprocessor()), 
            "",
            ptg.Button(f"{self.pp.button_color}train neural network", lambda *_: self.launch_trainer()), 
            "",
            ptg.Button(f"{self.pp.button_color}make predictions", lambda *_: self.launch_predictor()), 
            "",
            ptg.Button(f"{self.pp.button_color}conformance checking", lambda *_:self.launch_conformance()), 
            "",
            ptg.Button(f"{self.pp.button_color}back to menu", lambda *_: self.pp.switch_window(self.main_menu())), 
        ] 

        window = ptg.Window(*container, box="DOUBLE")
        window.center()
        return window

    def notify_project_creation(self, message, success):
        """
        function used to indicate that the new project name is valid
        as a result the window is switched to the menu for selecting the mode the currrent project is going to run in
        """ 
        if success: 
            container =ptg.Container( 
                ptg.Label(f"{message}"),
                "",
                ptg.Button(f"{self.pp.button_color}continue", lambda *_: self.pp.switch_window(self.select_mode())), 
                "",
                ptg.Button(f"{self.pp.button_color}Exit", lambda *_: self.pp.manager.stop())
            )
        # TODO: is this else part needed? Dont see it being called anywhere
        else: 
            container = ptg.Container( 
                ptg.Label(f"{message}!"),
                "",
                ptg.Button(f"{self.pp.button_color}back to menu", lambda *_: self.pp.switch_window(self.main_menu()))
            )


        window = ptg.Window(container, box="DOUBLE")
        window.center()
        return window 

    def handle_project_name_input(self):
        """
        exception if a new project is created with a name that is already used for another project in the projects directory
            -> user can return to this previous menu to create a new project with a different name

        If there is a valid input for the new project (unique name) then all of the necessary subdirectories are created where the 
        files needed for the different functionalities of the application are stored e.g. a subdirectory for the input log on which
        the RNN can then be trained
            -> user can then continue and select the mode in which he wants to work in the new project
        """
        name = self.project_name_input.value
        message = ""
        if name in os.listdir(f"{self.pp.state.projects_path}"):
            message = "directory already exits"
            container = ptg.Container(
                message, 
                "",
                ptg.Button("return", lambda *_: self.pp.switch_window(self.new_project_form()))
            )

            window = ptg.Window(container, box="DOUBLE")
            window.center()
            self.pp.switch_window(window)
            return

        message = f"directory created in path {os.getcwd()}/{self.pp.state.projects_path}/{name}"
        subdirectories = ["input_logs", "models", "petri_nets", "predictive_logs", "partial_traces", "decoded_dfs", "multiple_predictions_path"]
        os.mkdir(f"{os.getcwd()}/{self.pp.state.projects_path}/{name}")
        self.pp.state.current_project = name
        self.pp.state.input_logs_path  = f"{os.getcwd()}/{self.pp.state.projects_path}/{name}/input_logs"
        self.pp.state.models_path  = f"{os.getcwd()}/{self.pp.state.projects_path}/{name}/models"
        self.pp.state.petri_nets_path  = f"{os.getcwd()}/{self.pp.state.projects_path}/{name}/petri_nets"
        self.pp.state.predictive_logs_path  = f"{os.getcwd()}/{self.pp.state.projects_path}/{name}/predictive_logs"
        self.pp.state.partial_traces_path = f"{os.getcwd()}/{self.pp.state.projects_path}/{name}/partial_traces"
        self.pp.state.decoded_dfs_path = f"{os.getcwd()}/{self.pp.state.projects_path}/{name}/decoded_dfs"
        self.pp.state.multiple_predictions_path= f"{os.getcwd()}/{self.pp.state.projects_path}/{name}/multiple_predictions_path"
       
        for subdirectory in subdirectories: 
            os.mkdir(f"{os.getcwd()}/{self.pp.state.projects_path}/{name}/{subdirectory}")

        container =ptg.Container( 
            ptg.Label(f"{message}"),
            "",
            ptg.Button(f"{self.pp.button_color}continue", lambda *_: self.pp.switch_window(self.select_mode())), 
            "",
            ptg.Button(f"{self.pp.button_color}Exit", lambda *_: self.pp.manager.stop())
        )


        window = ptg.Window(container, box="DOUBLE")
        window.center()
        
        self.pp.switch_window(self.notify_project_creation(message, True))


    def handle_select_mode(self, mode: ProcessProphetMode):
        """
        indicates the previously selected mode the current project will be running in
        selected mode can be confirmed or changed if it was a missinput -> window either changes to previous menu or next menu to select further actions
        """
        self.pp.mode = mode
        container = [
            f"Currently in {mode.name} mode", 
            "", 
            ptg.Button(f"{self.pp.button_color}continue", lambda *_: self.pp.switch_window(self.select_manager())), 
            "",
            ptg.Button(f"{self.pp.button_color}back", lambda *_: self.pp.switch_window(self.select_mode()))
        ]

        window = ptg.Window(*container, box="DOUBLE")
        window.center()
        return window



    def select_mode(self):
        """
        menu to select wether the application should be run in base or advanced mode
        """ 
        container = [
            "Select a mode", 
            "",
            ptg.Button("quick", lambda *_: self.pp.switch_window(self.handle_select_mode(ProcessProphetMode.quick))),
            "",
            ptg.Button("advanced", lambda *_: self.pp.switch_window(self.handle_select_mode(ProcessProphetMode.advanced)))
        ]
        window = ptg.Window(*container, box="DOUBLE")
        window.center()
        return window


    def handle_project_selection(self):
        """
        checks if the selected project exists and passes over the directories that are needed for the different functionalities of the application
        e.g. "partial_traces" directory in order to make predictions

        The user is notified in the current window if the project is successfully selected and can then pursue further actions like selecting the mode
        of the application

        If the user enters a wrong file name the current window displays the error and the user can go back to the previous menu
        """

        projects = [project  for project in os.listdir(f"{self.pp.state.projects_path}")]
        name= self.input_select_project.value
        if name in projects: 

            self.pp.state.input_logs_path  = f"{os.getcwd()}/{self.pp.state.projects_path}/{name}/input_logs"
            self.pp.state.models_path  = f"{os.getcwd()}/{self.pp.state.projects_path}/{name}/models"
            self.pp.state.petri_nets_path  = f"{os.getcwd()}/{self.pp.state.projects_path}/{name}/petri_nets"
            self.pp.state.predictive_logs_path  = f"{os.getcwd()}/{self.pp.state.projects_path}/{name}/predictive_logs"
            self.pp.state.partial_traces_path = f"{os.getcwd()}/{self.pp.state.projects_path}/{name}/partial_traces"
            self.pp.state.decoded_dfs_path = f"{os.getcwd()}/{self.pp.state.projects_path}/{name}/decoded_dfs"
            self.pp.state.multiple_predictions_path= f"{os.getcwd()}/{self.pp.state.projects_path}/{name}/multiple_predictions_path"
            container =[  
                "Project selected successfully", 
                "",
                ptg.Button(f"{self.pp.button_color}continue", lambda *_: self.pp.switch_window(self.select_mode())), "",
                ptg.Button(f"{self.pp.button_color}back", lambda *_: self.pp.switch_window(self.main_menu()))
            ]
            window = ptg.Window(*container, box="DOUBLE")
            window.center()
            return window
        
        else: 
            container =  [ 
                "Project does not exist", 
                ptg.Button(f"{self.pp.button_color}back", lambda *_: self.pp.switch_window(self.load_existing_project()))
            ]
            window = ptg.Window(*container, box="DOUBLE")
            window.center()
            return window


    def load_existing_project(self):
        """
        user can load an existing project by entering the name of the existing project
        if intended the user can return to the main menu or quit the application
        """ 

        projects = [f"{project}"  for project in os.listdir(f"{self.pp.state.projects_path}")]

    
        self.input_select_project= ptg.InputField(projects[0],  prompt="enter a project name: ")
       
        left_container = ptg.Container(
            "[underline]Select a project", 
            "",
            self.input_select_project, 
            "",
            ptg.Button(f"{self.pp.button_color}Select", lambda *_: self.pp.switch_window(self.handle_project_selection())), 
            "", 
            ptg.Button(f"{self.pp.button_color}Back", lambda *_: self.pp.switch_window(self.main_menu()))
        )

        right_container= ptg.Container(
            "[underline]Existing projects", 
            *projects 
        )
        window = ptg.Window(ptg.Splitter(left_container,right_container), width = self.pp.window_width)
        #window = ptg.Window(*c, box="DOUBLE")
        window.center()
        return window

    def new_project_form(self):
        """
        user can create a new project and input a name for it
        if intended the user can return to the main menu or quit the application
        """

        self.project_name_input =  ptg.InputField("first Prophet", prompt="Project name: ")
        container =[ 
            ptg.Label(f"Create new project"),
            ptg.Label(f"current path: {os.getcwd()}/{self.pp.state.projects_path}"),
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
        """
        creates main menu for selecting a project to work on 
        """
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
