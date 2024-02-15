################################################################################
# Filename          : counter.py
# 
# Description       : This file is responsible for counting related process.
#
# Created by        : Brian Vergara Rosario on 01 July 2021
# Last Modified by  : Brian Vergara Rosario on 14 July 2021
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
# Counter Class (Start) 
# ==============================================================================

class Counter:
    """ This class is responsible for date/time related process.
    """
    
    # ==========================================================================
    # Constructor
    # ==========================================================================

    #####
    def __init__(self):
        """ Initialized Counter Class.
        """        

        # ----------------------------------------------------------------------
        # Public Variables
        # ----------------------------------------------------------------------

        ''' None '''

        # ----------------------------------------------------------------------
        # Private Variables
        # ----------------------------------------------------------------------
      
        # CPS Counter
        self.__cps_start            = None
        self.__cps_end              = None
        self.__cps                  = None
        self.__cps_counter          = None

        # CPS Limiter
        self.__cps_limit            = None
        self.__cps_limit_time_delta = None

        # Elapse Time
        self.__elapse_start         = None
        self.__elapse_end           = None
        self.__elapse_time          = None

        # ----------------------------------------------------------------------
        # Internal Method Calls
        # ----------------------------------------------------------------------
        
        ''' None '''

    # ==========================================================================
    # Public Methods
    # ==========================================================================

    #####
    def countdown(self, prompt: str, seconds: int):
        print()
        for i in range(seconds):               
            print(f"\r{prompt}: [{seconds - i}]", end='')            
            time.sleep(1)
        print("\n")  # New Line


    #####
    def get_cps(self):
        """ Cycles Per Second
        """
        if self.__cps_start == None:
            self.__cps_start    = time.time()
            self.__cps_counter  = 0
        else:
            self.__cps_end      = time.time()
            timediff            = self.__cps_end - self.__cps_start

            if timediff < 1.0:
                self.__cps_counter += 1
            else:
                self.__cps          = self.__cps_counter  # Get the final counter
                self.__cps_counter  = 0  # Reset
                self.__cps_start    = time.time()  # Start from the beginning

        return self.__cps


    #####
    def limit_cps(self, cycles_per_second):
        if self.__cps_limit == None :
            self.__cps_limit            = cycles_per_second
            self.__cps_limit_time_delta = 1./self.__cps_limit
        else:
            # time_now    = time.time()
            time.sleep(self.__cps_limit_time_delta)


    #####
    def elapse_time(self, milliseconds: int):
        if self.__elapse_time == None:
            self.__elapse_time = milliseconds
            self.__elapse_start = time.time()
        else:
            self.__elapse_end = time.time() - self.__elapse_start

            if self.__elapse_end >= (self.__elapse_time / 1000):
                self.__elapse_start = time.time()
                return True
            else:
                return False
        

    # ==========================================================================
    # Private Methods
    # ==========================================================================
    
    #####

# ==============================================================================
# Counter Class (End) 
# ==============================================================================



# ==============================================================================
# Program Entry Point (Start) 
# ==============================================================================

if __name__=="__main__":
    ''' This is optional '''
    ctr = Counter()
    ctr2 = Counter()

    is_running = True

    while is_running:
        if ctr.elapse_time(1500):
            print("1.5s has passed...")

        if ctr2.elapse_time(10000):
            print("10s has passed, exiting...")
            is_running = False

        ctr.limit_cps(60)

# ==============================================================================
# Program Entry Point (End) 
# ==============================================================================



################################################################################
# END OF FILE
################################################################################