################################################################################
# Filename          : datetimelog.py
# 
# Description       : This file is responsible for date/time related process.
#
# Created by        : Brian Vergara Rosario on 01 July 2021
# Last Modified by  : Brian Vergara Rosario on 01 September 2021
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
import time
import datetime
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
# DateTimeLog Class (Start) 
# ==============================================================================

class DateTimeLog:
    """ This class is responsible for date/time related process.
    """
    
    # ==========================================================================
    # Constructor
    # ==========================================================================

    #####
    def __init__(self):
        """ Initialized DateTimeLog Class.
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
    def get_datetime(self):
        """ Get the date + time

        Returns:
            Date + Time in this format: YYYY-MM-DD_HH-MM-SS
        """
        return datetime.datetime.now().strftime("%Y-%m-%d_%H:%M:%S")


    #####
    def get_datetime_ms_filename(self):
        """ Get the date + time

        Returns:
            A string formatted as: YYYYMMDDD_HHMMSS_mmm
        """
        return datetime.datetime.now().strftime("%Y%m%d_%H%M%S_%f")[:-3]


    #####
    def get_datetime_ms(self, milliseconds: int):
        """ Get the current date + time + milliseconds

        Args:
            milliseconds (int): 
                The length of the milliseconds from 1 to 6.

        Returns:
            Date + Time + Milliseconds in this format: YYYY-MM-DD_HH-MM-SS.mmmmmm
        """
        return self.__get_datetime_ms(milliseconds)

    # ==========================================================================
    # Private Methods
    # ==========================================================================
    
    #####
    def __get_datetime_ms(self, milliseconds: int):
        ms = 0
        if milliseconds < 6:
            ms = 6 - milliseconds
            return datetime.datetime.now().strftime("%Y-%m-%d_%H:%M:%S.%f")[:-(ms)]
        else:
            return datetime.datetime.now().strftime("%Y-%m-%d_%H:%M:%S.%f")

# ==============================================================================
# DateTimeLog Class (End) 
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