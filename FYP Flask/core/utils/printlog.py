################################################################################
# Filename          : printlog.py
# 
# Description       : This file is responsible for displaying colored messages
#                     on the console.  It is primarily used by the logger, but 
#                     it can also be used directly.
#
# Created by        : Brian Vergara Rosario on 11 June 2021
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

from colored import fg, bg, attr

# ------------------------------------------------------------------------------
# 1st Party Libraries (Our Own Libraries)
# ------------------------------------------------------------------------------

from core.utils.colortable import Fore, Back, Style



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

LOG_TYPE            = ("INFO", "DEBUG", "WARNING", "CRITICAL", "ERROR")  # Tuple
LOG_TYPE_CUSTOM     = ("RESULT", "OTHERS")  # Tuple

# ------------------------------------------------------------------------------
# Global Variables
# ------------------------------------------------------------------------------

''' None '''



################################################################################
# CLASSES & METHODS, FUNCTIONS, ROUTES
################################################################################

# ==============================================================================
# PrintLog Class (Start) 
# ==============================================================================

class PrintLog:
    """ This class is responsible for displaying colored messages on 
        the console.  It is primarily used by the logger, but it can alose 
        be used directly.
    """
    
    # ==========================================================================
    # Constructor
    # ==========================================================================

    #####
    def __init__(self):
        """ Initialized PrintLog Class.
        """

        # ----------------------------------------------------------------------
        # Public Variables
        # ----------------------------------------------------------------------

        ''' None '''

        # ----------------------------------------------------------------------
        # Private Variables
        # ----------------------------------------------------------------------

        ''' None '''

        # ----------------------------------------------------------------------
        # Internal Method Calls
        # ----------------------------------------------------------------------
    
        # colorama.init()

    # ==========================================================================
    # Public Methods
    # ==========================================================================
   
    #####
    def print_log(self, type: str, message: str):
        """ Print a log message into the screen with color text and background.
            The color depends on what type of log you wanted to print.
        Args:
            type (str): 
                Type must be one of these values: INFO, DEBUG, WARNING, CRITICAL, ERROR or any custom type such as RESULT.
            message (str): 
                The log message.
        """

        if type in LOG_TYPE:
            if type.upper() == LOG_TYPE[0]:  # INFO
                self.__print_to_screen(Fore.BLACK, Back.GREEN, f"INFO: {message}")
            elif type.upper() == LOG_TYPE[1]:  # DEBUG
                self.__print_to_screen(Fore.BLACK, Back.CYAN, f"DEBUG: {message}")
            elif type.upper() == LOG_TYPE[2]:  # WARNING
                self.__print_to_screen(Fore.BLACK, Back.LIGHT_YELLOW, f"WARNING: {message}")
            elif type.upper() == LOG_TYPE[3]:  # CRITICAL
                self.__print_to_screen(Fore.BLACK, Back.LIGHT_MAGENTA, f"CRITICAL: {message}")
            elif type.upper() == LOG_TYPE[4]:  # ERROR
                self.__print_to_screen(Fore.WHITE, Back.RED, f"ERROR: {message}")
            else:
                pass
        elif type in LOG_TYPE_CUSTOM:
            if type.upper() == LOG_TYPE_CUSTOM[0]:  # RESULT
                print()
                self.__print_to_screen(Fore.BLACK, Back.WHITE, f"RESULT: {message}")
                print()
            else:
                pass
        '''
        Fore: BLACK, RED, GREEN, YELLOW, BLUE, MAGENTA, CYAN, WHITE, RESET.
        Back: BLACK, RED, GREEN, YELLOW, BLUE, MAGENTA, CYAN, WHITE, RESET.
        Style: DIM, NORMAL, BRIGHT, RESET_ALL
        '''
    

    # ==========================================================================
    # Private Methods
    # ==========================================================================
    
    #####
    def __print_to_screen(self, foreground_color: Fore, back_ground_color: Back, message: str):    
        """ The actual method that print a log messages on the screen with colors.

        Args:
            foreground_color (Fore): 
                Fore: BLACK, RED, GREEN, YELLOW, BLUE, MAGENTA, CYAN, WHITE, RESET.
            back_ground_color (Back):
                Back: BLACK, RED, GREEN, YELLOW, BLUE, MAGENTA, CYAN, WHITE, RESET.
            message (str): 
                Style: DIM, NORMAL, BRIGHT, RESET_ALL
        """

        # print(foreground_color + back_ground_color + f" {message} " + Style.RESET_ALL)
        #print(foreground_color + back_ground_color + f" {message} ")
        print(f"{fg(foreground_color)}{bg(back_ground_color)} {message} {attr(Style.RESET)}")

# ==============================================================================
# PrintLog Class (End) 
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