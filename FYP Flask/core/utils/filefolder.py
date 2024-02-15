################################################################################
# Filename          : filefolder.py
# 
# Description       : The file is responsible for checking and creating folders.
#
# Created by        : Brian Vergara Rosario on 10 June 2021
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

# ------------------------------------------------------------------------------
# Global Variables
# ------------------------------------------------------------------------------

''' None '''



################################################################################
# CLASSES & METHODS, FUNCTIONS, ROUTES
################################################################################

# ==============================================================================
# FileFolder Class (Start) 
# ==============================================================================

class FileFolder:
    """ This class is responsible for for checking and creating folders.
    """
    
    # ==========================================================================
    # Constructor
    # ==========================================================================

    #####
    def __init__(self):
        """ Initialized FileFolder Class.
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
    
        ''' None '''

    # ==========================================================================
    # Public Methods
    # ==========================================================================
   
    #####
    def is_present(self, path: str):
        """ Check if a file or folder exist.

        Args:
            path (str): 
                The path of the file or folder. 
                e.g.: "C:/temp/this_folder" or "C:/temp/hello_world.txt"

        Returns:
            boolean: 
                True if a file or folder exists, or False if not.
        """     

        fix_path = Path(path)
        if os.path.exists(fix_path):
            return True
        else:
            return False


    #####
    def create_folder(self, folder_name: str):
        """ Create a folder.

        Args:
            folder_name (str): 
                The name of the folder with path. 
                e.g. "C:/temp/this_folder" = will create the "this_folder" folder inside the "C:/temp" directory.

        Returns:
            boolean: 
                True if a folder was successfully created, or False if it fails.
        """

        fix_path = Path(folder_name)    
        try:            
            os.makedirs(fix_path)
            return True
        except Exception as e:
            return False        

    # ==========================================================================
    # Private Methods
    # ==========================================================================

    '''None'''

# ==============================================================================
# FileFolder Class (End) 
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