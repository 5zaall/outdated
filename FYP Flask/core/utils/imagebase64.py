################################################################################
# Filename          : imagebase64.py
# 
# Description       : This file is responsible for encoding images into a
#                     readable string for ease of transmission over the network.
#                     The encoded images can then be decoded back to its binary
#                     form.
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

import base64
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
# ImageBase64 Class (Start) 
# ==============================================================================

class ImageBase64:
    """ This class is responsible for encoding images into a readable string for
        ease of transmission over the network.  The encoded images can then be 
        decoded back to its binary form.
    """
    
    # ==========================================================================
    # Constructor
    # ==========================================================================

    #####
    def __init__(self):
        """ Initialized ImageBase64 Class.
        """        

        # ----------------------------------------------------------------------
        # Public Variables
        # ----------------------------------------------------------------------

        ''' None '''

        # ----------------------------------------------------------------------
        # Private Variables
        # ----------------------------------------------------------------------
      
        self.__image_base64     = None    

        # ----------------------------------------------------------------------
        # Internal Method Calls
        # ----------------------------------------------------------------------

        ''' None '''

    # ==========================================================================
    # Public Methods
    # ==========================================================================
   
    #####
    def __repr__(self):
        return self.__image_base64


    #####
    def b64_string(self, string: str):
        self.__image_base64 = string
        

    #####
    def encode(self, filename: str) -> str:
        """ Encode a given image to Base64 string.

        Args:
            filename (str): 
                The image file path and filename of the input image.

        Returns:
            str: The Base64 string of the given image.
        """

        from core.utils.logger import Logger
        log = Logger()

        try:
            with open(filename, "rb") as image_file:
                try:
                    self.__image_base64 = base64.b64encode(image_file.read()).decode("utf-8")
                    log.module_log("INFO", "image_encdec.py", f"open() | {filename} has been encoded to Base64.")
                    return self.__image_base64                    
                except Exception as e:  
                    log.module_log("ERROR", "image_encdec.py", f"open() | Error encoding {filename} to Base64.  Exception: {e}")
        except Exception as e:
            log.module_log("ERROR", "image_encdec.py", f"open() | {e}")


    #####
    def decode(self, filename: str):
        """ Decode a given image from Base64.

        Args:
            filename (str):
                The image file path and filename of the output image.
        """

        from core.utils.logger import Logger
        log = Logger()

        try:
            with open(filename, "wb") as image_file:
                try:
                    self.__image_base64 = base64.decodebytes(self.__image_base64.encode("utf-8"))
                    image_file.write(self.__image_base64)
                    log.module_log("INFO", "image_encdec.py", f"decode() | {filename} has been decoded from Base64.")
                except Exception as e:
                    log.module_log("ERROR", "image_encdec.py", f"decode() | Error decoding {filename} to Base64.  Exception: {e}")            
        except Exception as e:
            log.module_log("ERROR", "image_encdec.py", f"decode() | {e}")            


    # ==========================================================================
    # Private Methods
    # ==========================================================================
    
    ''' None '''

# ==============================================================================
# ImageBase64 Class (End) 
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