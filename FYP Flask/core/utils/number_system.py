################################################################################
# Filename          : number_system.py
# 
# Description       : This file is responsible for converting a number system to
#                     another number system.
#
# Created by        : Brian Vergara Rosario on 15 July 2021
# Last Modified by  : Brian Vergara Rosario on 19 August 2021
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
import multiprocessing
import subprocess
import time
import threading
from pathlib import Path


# Below is a simple fix to address the "ModuleNotFoundError" when importing
# "appcore.modulename" modules from a directory other than the parent directory.
#
# Parameters:   '../'   means 1 folder above.
#               '../..' means 2 folders above. 
sys.path.insert(1, os.path.join(sys.path[0], '../../..'))

# ------------------------------------------------------------------------------
# 3rd Party Libraries
# ------------------------------------------------------------------------------

import serial

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
APPCORE_ROOT = Path(__file__).resolve().parents[3]

# ------------------------------------------------------------------------------
# Global Variables
# ------------------------------------------------------------------------------

''' None '''



################################################################################
# CLASSES & METHODS, FUNCTIONS, ROUTES
################################################################################

# ==============================================================================
# NumberSystem Class (Start) 
# ==============================================================================

class NumberSystem:
    """ This class is responsible for converting a number system to another
        number system.
    """
    
    # ==========================================================================
    # Constructor
    # ==========================================================================

    #####
    def __init__(self):
        """ Initialized the NumberSystem Class.
        """

        # ----------------------------------------------------------------------
        # Public Variables
        # ----------------------------------------------------------------------
        
        ''' Name '''

        # ----------------------------------------------------------------------
        # Private Variables
        # ----------------------------------------------------------------------

        ''' Name '''

        # ----------------------------------------------------------------------
        # Internal Method Calls
        # ----------------------------------------------------------------------
    
        ''' None '''

    # ==========================================================================
    # Public Methods
    # ==========================================================================
   

    #####
    def bin_to_dec(self, bin_value: str):
        temp_dec = self.hex_to_dec(self.bin_to_hex(bin_value))
        return temp_dec


    #####
    def bin_to_oct(self, bin_value: str):
        pass


    ##### Done
    def bin_to_hex(self, bin_value: str):
        temp_hex = hex(int(bin_value, 2))[2:]
        return temp_hex.upper()
    

    #####
    def dec_to_bin(self, dec_value: int):
        pass


    #####
    def dec_to_oct(self, dec_value: int):
        pass


    ##### Done
    def dec_to_hex(self, dec_value: int):
        dec2hex     = hex(dec_value)
        dec2hex_fix = dec2hex[2::]
        temp_hex = ""

        if len(dec2hex_fix) == 3:  # Fix string length.
            temp_hex = "0x0" + dec2hex_fix
        else:
            temp_hex = dec2hex

        temp_list       = []
        split_length    = 2
        for i in range(0, len(temp_hex), split_length):
            temp_list.append("0x" + temp_hex[::1][i:i+split_length].upper())
        
        temp_list.pop(0)  # Remove the first element from the list.

        return temp_list


    #####
    def oct_to_bin(self, oct_value: int):
        pass


    #####
    def oct_to_dec(self, oct_value: int):
        pass


    #####
    def oct_to_hex(self, oct_value: int):
        pass


    ##### Done
    def hex_to_bin(self, hex_value, return_length: int):
        """ Convert Hexadecimal to Binary.

        Args:
            hex_value ([type]):
                The hexadecimal value that needs to be converted.
            return_length (int): 
                The length of the binary string that will be returned.
                E.g. If return_length = 8, it will return "00000000". 
                If return_length = 4, will return "0000".

        Returns:
            [type]: [description]
        """

        temp_bin = None

        # print(f"number_system.py > hex_to_bin() > hex_value: {hex_value}")        

        if isinstance(hex_value, str):
            temp_bin = "{0:b}".format(int(hex_value, 16))        

        elif isinstance(hex_value, int):
            temp_bin = "{0:b}".format(int(str(hex_value), 16))

        elif isinstance(hex_value, bytes):
            # print(f"hex_value: {hex_value}")
            if hex_value == b'':                
                hex_value = b"0"
                temp_bin = "{0:b}".format(int(str(hex_value.decode("UTF-8")), 16))
            else:
                temp_bin = "{0:b}".format(int(str(hex_value.decode("UTF-8")), 16))
            # print(f"number_system.py > hex_to_bin() > temp_bin: {temp_bin}")  
        
        # print(f"number_system: temp_bin: {temp_bin}")

        # Fix Prefix Zeros in Binary
        if len(temp_bin) < return_length:
            temp_bin_len        = len(temp_bin)
            temp_bin_len_diff   = return_length - temp_bin_len

            temp_bin_fix        = "0" * temp_bin_len_diff + temp_bin
            temp_bin = temp_bin_fix

        return temp_bin


    #####
    def hex_to_oct(self, hex_value: str):
        pass


    ##### Done
    def hex_to_dec(self, hex_value):

        temp_dec = None

        if isinstance(hex_value, str):
            temp_dec = int(hex_value, 16)

        elif isinstance(hex_value, int):
            temp_dec = int(str(hex_value), 16)
        
        return temp_dec   


    #####
    def hex_to_dec_multi(self, hex_string: bytes):
        temp_hex        = hex_string.hex()
        temp_list       = []
        split_length    = 2
        for i in range(0, len(temp_hex), split_length):
            temp_list.append(int(temp_hex[::1][i:i+split_length], 16))

        #print(f"temp_list: {temp_list}")

        return temp_list


    #####
    def ascii_to_bin(self, ascii: str):
        pass


    #####
    def ascii_to_dec(self, ascii: str):
        pass


    #####
    def ascii_to_oct(self, ascii: str):
        pass


    ##### Done
    def ascii_to_hex(self, ascii: str):
        temp_hex = []
    
        for i in ascii:
            temp_hex.append(hex(ord(i)))
    
        return temp_hex


    ##### Done
    def xor_bin(self, bin_value1: str, bin_value2: str, return_length: int):
        temp_val1   = int(bin_value1, 2)
        temp_val2   = int(bin_value2, 2)

        temp_xor    = temp_val1 ^ temp_val2
        temp_bin    = "{0:b}".format(temp_xor)

        # Fix Length
        bin_len = len(temp_bin)
        if bin_len < return_length:
            for i in range(return_length - bin_len):
                temp_bin2 = "0" + temp_bin
                temp_bin = temp_bin2

        return temp_bin        

    # ==========================================================================
    # Private Methods
    # ==========================================================================

    ''' None '''

# ==============================================================================
# NumberSystem Class (End) 
# ==============================================================================



# ==============================================================================
# Program Entry Point (Start) 
# ==============================================================================

if __name__=="__main__":
    ''' This is optional '''
    
    num = NumberSystem()

    base2       = 10100110
    base10      = 60000
    base16      = "0x5C"
    frame_data  = None

    #print(f"Hex: {base16} -> Bin: {num.hex_to_bin(base16)}")
    #print(f"Hex: {base16} -> Dec: {num.hex_to_dec(base16)}")
    # print(f"Dec: {base10} -> Hex: {num.dec_to_hex(base10)}")

    h = num.dec_to_hex(base10)
    print(len(h))



    if len(h) == 1:     # 1 byte
        frame_data = [0x87, 0xC0, ]
    elif len(h) == 2:   # 2 bytes
        pass
    else:
        pass

# ==============================================================================
# Program Entry Point (End) 
# ==============================================================================



################################################################################
# END OF FILE
################################################################################