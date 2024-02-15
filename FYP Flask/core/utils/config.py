################################################################################
# Filename          : config.py
# 
# Description       : This file is responsible for reading configuration files.
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

from configparser import ConfigParser

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

# ------------------------------------------------------------------------------
# Global Variables
# ------------------------------------------------------------------------------

''' None '''



################################################################################
# CLASSES & METHODS, FUNCTIONS, ROUTES
################################################################################

# ==============================================================================
# Config Class (Start) 
# ==============================================================================

class Config:
    """ This class is responsible for reading configuration files.
    """
    
    # ==========================================================================
    # Constructor
    # ==========================================================================

    #####
    def __init__(self):
        """ Initialized the Config Class.
        """

        # ----------------------------------------------------------------------
        # Public Variables
        # ----------------------------------------------------------------------

        ''' None '''

        # ----------------------------------------------------------------------
        # Private Variables
        # ----------------------------------------------------------------------

        self.__parser       = ConfigParser()
        self.__value        = None
        self.__ini          = None

        # ----------------------------------------------------------------------
        # Internal Method Calls
        # ----------------------------------------------------------------------
    
        ''' None '''

    # ==========================================================================
    # Public Methods
    # ==========================================================================
   
    #####
    def open(self, ini_file: str):
        """ Open an ini configuration file.

        Args:
            ini_file (str):
                The filename of the ini file with absolute or relative path.

        Returns:
            bool
        """

        try:
            self.__ini = ini_file
            if self.__parser.read(self.__ini):
                return True
            else:
                return False
        except Exception as e:
            return False       


    #####
    def read(self, section: str, key: str):
        """ Read a [key] from the ini file.

        Args:
            section (str):
                The ini [section] name.
            key (str):
                The ini  [key] name.

        Returns:
            Key Value or False
        """

        try:
            self.__value = self.__parser.get(section, key)
            return self.__value
        except Exception as e:
            return False         
           

    # ==========================================================================
    # Private Methods
    # ==========================================================================

    ''' None '''

# ==============================================================================
# Config Class (End) 
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