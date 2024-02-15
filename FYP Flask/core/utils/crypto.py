################################################################################
# Filename          : crypto.py
# 
# Description       : This file is responsible for encrypting and decrypting
#                     a string of text.  It is mainly used for passwords.
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

from cryptography.fernet import Fernet
from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2HMAC

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
# Crypto Class (Start) 
# ==============================================================================

class Crypto:
    """ The class is responsible for encrypting and decrypting a string of text.
        It is mainly used for passwords.
    """
    
    # ==========================================================================
    # Constructor
    # ==========================================================================

    #####
    def __init__(self):
        """ Initialized Crypto Class.
        """

        # ----------------------------------------------------------------------
        # Public Variables
        # ----------------------------------------------------------------------

        ''' None '''

        # ----------------------------------------------------------------------
        # Private Variables
        # ----------------------------------------------------------------------

        self.__password     = b"CZRjTnsvukznTfWpf9YeUrXRuk8p6sHmp3d3AvR3YNAD8UQ2kYbSwtfw9VSTgAqZ"
        self.__salt         = b"xKDkxpLCNPjYdC73ZMaps3ZRtXXaqfYKVx4rPgD4vwFHH59csmWGq7Q6qty9rzcU"

        # PBKDF2HMAC Configuration
        self.__algorithms   = hashes.SHA512()
        self.__length       = 32        
        self.__iterations   = 100000

        # Initialize PBKDF2HMAC
        self.__kdf          = PBKDF2HMAC(
                                algorithm   = self.__algorithms,
                                length      = self.__length,
                                salt        = self.__salt,
                                iterations  = self.__iterations)

        self.__key          = base64.urlsafe_b64encode(self.__kdf.derive(self.__password))
        self.__fernet       = Fernet(self.__key)            

        # ----------------------------------------------------------------------
        # Internal Method Calls
        # ----------------------------------------------------------------------
    
        ''' None '''

    # ==========================================================================
    # Public Methods
    # ==========================================================================
   
    #####
    def enc(self, string: str):
        return self.__fernet.encrypt(string.encode()).decode("utf-8")  # Convert Bytes to String


    #####
    def dec(self, string: str):
        if type(string) is bytes:
            return self.__fernet.decrypt(string).decode("utf-8")  # Convert Bytes to String
        else:
            return self.__fernet.decrypt(string.encode()).decode("utf-8")  # Convert Bytes to String


    #####
    def password_match(self, password: str, encrypted_password: str):
        try:
            if password == self.dec(encrypted_password):
                return True
            else:
                return False
        except:
            return False   
           

    # ==========================================================================
    # Private Methods
    # ==========================================================================

    ''' None '''

# ==============================================================================
# Crypto Class (End) 
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