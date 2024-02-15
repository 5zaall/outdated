################################################################################
# Filename          : logger.py
# 
# Description       : This file is responsible for displaying log messages into
#                     the console, as well as writing them into a log files.
#
# Created by        : Brian Vergara Rosario on 08 June 2021
# Last Modified by  : Brian Vergara Rosario on 05 July 2021
#
# Copyright         : © 2021 N8XT Aerospace Pte. Ltd.  All Rights Reserved.
################################################################################



################################################################################
# IMPORTS
################################################################################

# ------------------------------------------------------------------------------
# Standard Python Libraries
# ------------------------------------------------------------------------------
import datetime
import os
import sys
from pathlib import Path

# Below is a simple fix to address the "ModuleNotFoundError" when importing
# "appcore.modulename" modules from a directory other than the parent directory.
#
# Parameters:   '../'   means 1 folder above.
#               '../..' means 2 folders above. 
sys.path.insert(1, os.path.join(sys.path[0], '../..'))

# ------------------------------------------------------------------------------
# 3rd Party Libraries
# ------------------------------------------------------------------------------

''' None '''

# ------------------------------------------------------------------------------
# 1st Party Libraries (Our Own Libraries)
# ------------------------------------------------------------------------------

''' None '''



################################################################################
# GLOBALS
################################################################################

# ------------------------------------------------------------------------------
# Global Constants
# ------------------------------------------------------------------------------

# Below is the constants for AppCORE root folder.
# 
# Paremeters:   '.parents[x]' the x means depth of the folder.
APPCORE_ROOT = Path(__file__).resolve().parents[2]

LOG_TYPE = ("INFO", "DEBUG", "WARNING", "CRITICAL", "ERROR")  # Tuple

# ------------------------------------------------------------------------------
# Global Variables
# ------------------------------------------------------------------------------

''' None '''



################################################################################
# CLASSES & METHODS, FUNCTIONS, ROUTES
################################################################################

# ==============================================================================
# Logger Class (Start) 
# ==============================================================================

class Logger:
    """ This class is responsible for displaying log messages into the console,
        as well as writing them into a log files.
    """

    #####
    def __init__(self):        
        """ Initialized Logger Class.
        """

        # ----------------------------------------------------------------------
        # Public Variables
        # ----------------------------------------------------------------------

        '''None'''

        # ----------------------------------------------------------------------
        # Private Variables
        # ----------------------------------------------------------------------

        # Logger Message
        self.__message                  = None  # The log message.
        self.__log_type                 = None  # INFO, DEBUG, ERROR, WARNING, CRITICAL          
        self.__module_name              = None  # Add the source filename (e.g. logger.py) on the log file.                                                
        self.__log_name                 = None  # The name of the log file.  It is used by app_log() and app_log_multi().

        # Logger Settings                
        self.__ini                      = f"{APPCORE_ROOT}\config\logger.ini"        
        self.__author                   = None  # The author of the log message, either APP, APP_MULTI, MODULE, or MODULE_MULTI

        # Logger Settings for App Logs
        self.__log_root                 = f"{APPCORE_ROOT}\logs\\"        
        self.__app_log_to_console       = None  # Enable or disable showing the log messages in the console.
        self.__app_log_to_file          = None  # Enable or disable writing the log messages in a log file.

        # Logger Settings for Module Logs
        self.__log_modules              = f"{APPCORE_ROOT}\logs\modules\\"
        self.__module_log_to_console    = None  # Enable or disable showing the log messages in the console.
        self.__module_log_to_file       = None  # Enable or disable writing the log messages in a log file.

        self.__ini_parsed               = False # True = logger.ini has been read, or False = not

        # ----------------------------------------------------------------------
        # Internal Method Calls
        # ---------------------------------------------------------------------

        self.__check_log_folders()  # Check Folders
        self.__read_config()  # Read INI file              
    
    # ==========================================================================
    # Public Methods
    # ==========================================================================
   
    #####
    def config(self, log_to_console: bool=True, log_to_file: bool=True):
        self.__app_log_to_console       = log_to_console        
        self.__app_log_to_file          = log_to_file
        self.__module_log_to_console    = log_to_console
        self.__module_log_to_file       = log_to_file


    #####
    def app_log(self, type: str, log_name: str, message: str):
        """ Used in writing a log messages in a single log file.  
            The log file is saved inside the folder [ROOT > logs].

        Args:
            type (str): 
                Type must be one of these values: INFO, DEBUG, ERROR, WARNING, CRITICAL
            log_name (str): 
                The filename of the log file.  It is used as the filename's prefix of the log file.
            message (str): 
                The log message.
        """

        self.__author           = "APP"
        self.__log_type         = type.upper()
        self.__log_name         = log_name
        # self.__message          = f"{self.__log_type} | {message}"
        self.__message          = message
        self.__process_log()


    #####
    def app_log_multi(self, type: str, log_name: str, message: str):
        """ Used in writing log messages in multiple log files segregated based on types.
            The log files are saved inside the folder [ROOT > logs].

        Args:
            type (str): 
                Type must be one of these values: INFO, DEBUG, ERROR, WARNING, CRITICAL
            log_name (str): 
                The filename of the log file.  It is used as the filename's prefix of the log files.
            message (str): 
                The log message.
        """      

        self.__author           = "APP_MULTI"
        self.__log_type         = type.upper()
        self.__log_name         = log_name
        # self.__message          = f"{self.__log_type} | {message}"
        self.__message          = message
        self.__process_log()


    #####
    def module_log(self, type: str, module_name: str, message: str):
        """ Used in writing a log messages in a single log file.  
            It is primarily used for writing a module's log messages.
            The log file is saved inside the folder [ROOT > logs > modules].

        Args:
            type (str):
                Type must be one of these values: INFO, DEBUG, WARNING, CRITICAL, ERROR
            module_name (str):
                The filename of the module, e.g. logger.py.  It is used as the filename's prefix of the log file.
            message (str):
                The log message.
        """         

        self.__author           = "MODULE"
        self.__log_type         = type.upper()
        self.__module_name      = module_name
        # self.__message          = f"{self.__log_type} | {self.__module_name} | {message}"
        self.__message          = f"{self.__module_name} | {message}"
        self.__process_log()


    #####
    def module_log_multi(self, type: str, module_name: str, message: str):
        """ Used in writing log messages in multiple log files segregated based on types.  
            It is primarily used for writing a module's log messages.
            The log files is saved inside the folder [ROOT > logs > modules].

        Args:
            type (str): 
                Type must be one of these values: INFO, DEBUG, ERROR, WARNING, CRITICAL
            module_name (str): 
                The filename of the module, e.g. logger.py.  It is used as the filename's prefix of the log file.
            message (str): 
                The log message.
        """      

        self.__author           = "MODULE_MULTI"
        self.__log_type         = type.upper()
        self.__module_name      = module_name
        # self.__message          = f"{self.__log_type} | {self.__module_name} | {message}"
        self.__message          = f"{self.__module_name} | {message}"
        self.__process_log()

    # ==========================================================================
    # Private Methods
    # ==========================================================================

    #####
    def __process_log(self):
        """ This method was called by app_log(), app_log_multi(), module_log(), and module_log_multi() when
            displaying and/or writing log messages into the console or log file.

        Raises:
            ValueError: Enter one of the following types to prevent raising an error: INFO, DEBUG, ERROR, WARNING, CRITICAL
        """

        if self.__log_type in LOG_TYPE:
            # self.__log_to_console()
            self.__log_to_console_color()
            self.__log_to_file()
        else:
            self.module_log("ERROR", "logger.py", f"Unknown log type [{self.__log_type}].  Please enter any of the following value: {LOG_TYPE}")
            raise ValueError
    
    
    #####
    def __log_to_file(self):
        """ This method preformat the filename and messages before writing the log messages into the log files.
        """

        current_datetime_log_message = datetime.datetime.now().strftime("%Y-%m-%d_%H:%M:%S.%f")[:-3]
        current_datetime_log_filename = datetime.datetime.now().strftime("%Y-%m-%d")     

        if self.__author == "APP":
            if self.__app_log_to_file:
                filename = f"{self.__log_root}{self.__log_name}_{current_datetime_log_filename}.log"
                # message = f"{current_datetime_log_message} | {self.__message}"
                message = f"{current_datetime_log_message} | {self.__log_type} | {self.__message}"
                self.__write_to_file(filename, message)  # Write to log file.

        if self.__author == "APP_MULTI":
            if self.__app_log_to_file:
                filename = f"{self.__log_root}{self.__log_name}_{self.__log_type}_{current_datetime_log_filename}.log"
                # message = f"{current_datetime_log_message} | {self.__message}"
                message = f"{current_datetime_log_message} | {self.__log_type} | {self.__message}"
                self.__write_to_file(filename, message)  # Write to log file.

        elif self.__author == "MODULE":
            if self.__module_log_to_file:
                module_folder = f"{self.__log_modules}\\"
                if self.__check_log_folder(module_folder):
                    filename = f"{module_folder}{self.__module_name[:-3]}_{current_datetime_log_filename}.log"
                    # message = f"{current_datetime_log_message} | {self.__message}"
                    message = f"{current_datetime_log_message} | {self.__log_type} | {self.__message}"
                    self.__write_to_file(filename, message)  # Write to log file.
                else:
                    self.module_log("ERROR", "logger.py", f"__log_to_file | Cannot find/create the module's log folder.")

        elif self.__author == "MODULE_MULTI":
            if self.__module_log_to_file:
                module_folder = f"{self.__log_modules}{self.__module_name[:-3]}\\"
                if self.__check_log_folder(module_folder):
                    filename = f"{module_folder}{self.__log_type}_{current_datetime_log_filename}.log"
                    # message = f"{current_datetime_log_message} | {self.__message}"
                    message = f"{current_datetime_log_message} | {self.__log_type} | {self.__message}"
                    self.__write_to_file(filename, message)  # Write to log file.
                else:
                    self.module_log("ERROR", "logger.py", f"__log_to_file | Cannot find/create the module's log folder.")
        else:
            pass


    #####
    def __write_to_file(self, filename: str, message: str):
        """ This method is the one who actually writes the log messages into the log files.

        Args:
            filename (str): 
                The log file's filename.
            message (str): 
                The log messages.
        """

        try:
            logfile = open(filename, "a")
            logfile.write(f"{message}\n")
        except Exception as e:
            self.module_log("ERROR", "logger.py", f"__write_to_file | {e}")
        finally:
            logfile.close()


    #####
    def __log_to_console(self):
        """ This method is the one who actually displays the log messages into the console.
        """

        if self.__author == "APP" or self.__author == "APP_MULTI":
            if self.__app_log_to_console:
                # print(f"{self.__message}")
                print(f"{self.__log_type} | {self.__message}")
        elif self.__author == "MODULE" or self.__author == "MODULE_MULTI":
            if self.__module_log_to_console:
                # print(f"{self.__message}")
                print(f"{self.__log_type} | {self.__message}")
        else:
            pass


    #####
    def __log_to_console_color(self):
        """ This method is the one who actually displays the log messages into the console, with color of course =)
        """

        from core.utils.printlog import PrintLog
        color_log = PrintLog()

        if self.__author == "APP" or self.__author == "APP_MULTI":
            if self.__app_log_to_console:
                # print(f"{self.__message}")
                color_log.print_log(self.__log_type, self.__message)
        elif self.__author == "MODULE" or self.__author == "MODULE_MULTI":
            if self.__module_log_to_console:
                # print(f"{self.__message}")
                color_log.print_log(self.__log_type, self.__message)
        else:
            pass


    #####
    def __read_config(self):
        """ This method reads the logger configuration from the [logger.ini] file located inside the [ROOT > config] folder.

        Returns:
            [type]: bool
        """

        from core.utils.config import Config
        from core.utils.string2bool import StringToBool

        ini = Config()
        s2b = StringToBool()

        if ini.open(self.__ini):
            try:
                # Read APPCORE Settings
                self.__app_log_to_console       = s2b.convert(ini.read("app", "log_to_console"))
                self.__app_log_to_file          = s2b.convert(ini.read("app", "log_to_file"))                     

                # Read MODULES Settings
                self.__module_log_to_console    = s2b.convert(ini.read("modules", "log_to_console"))
                self.__module_log_to_file       = s2b.convert(ini.read("modules", "log_to_file"))

                # self.module_log("INFO", "logger.py", f"__read_config() | The ini file [{self.__ini}] has been read successfully.")
                return True
                                
            except Exception as e:
                self.module_log("ERROR", "logger.py", f"__read_config() | {e}")
                return False
        else:
            self.__module_log_to_console        = True
            self.__module_log_to_file           = True
            self.module_log("ERROR", "logger.py", f"__read_config | Cannot read the configuration file [logger.ini].")
            return False


    #####
    def __check_log_folder(self, folder_name: str):
        """ This method is check the folder if it exists.  It will try to create a new one if the folder does not exists.

        Args:
            folder_name (str): 
                The folder name.

        Returns:
            [type]: bool
        """

        from core.utils.filefolder import FileFolder
        ff = FileFolder()

        if ff.is_present(folder_name):
            return True
        else:
            if ff.create_folder(folder_name):
                # self.module_log("INFO", "logger.py", f"__check_log_folder() | Folder [{folder_name}] has been created.")
                return True
            else:
                self.module_log("ERROR", "logger.py", f"__check_log_folder() | Failed creating folder [{folder_name}].")
                return False


    #####
    def __check_log_folders(self):
        """ This method will try to check if the default log folders exists 
            and will try to create one if it's does not exist.
        """
        
        self.__check_log_folder(self.__log_root)
        self.__check_log_folder(self.__log_modules)

# ==============================================================================
# Logger Class (End) 
# ==============================================================================



# ==============================================================================
# Program Entry Point (Start) 
# ==============================================================================

if __name__=="__main__":
    ''' This is optional '''

# ==============================================================================
# Program Entry Point (End) 
# ==============================================================================



################################################################################
# END OF FILE
################################################################################