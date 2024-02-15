################################################################################
# Filename          : keyboard.py
# 
# Description       : This file is responsible for non-blocking reading of the
#                     keyboard.
#
# Created by        : Brian Vergara Rosario on 01 July 2021
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

if os.name == 'nt':  # Windows
    import msvcrt

else:  # Linux, macOS
    import sys
    import termios
    import atexit
    from select import select

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
# Keyboard Class (Start) 
# ==============================================================================

class Keyboard:
    """ This class is responsible for non-blocking reading of the keyboard.
    """
    
    # ==========================================================================
    # Constructor
    # ==========================================================================

    #####
    def __init__(self):
        """ Initialized Keyboard Class.
        """        

        # ----------------------------------------------------------------------
        # Public Variables
        # ----------------------------------------------------------------------

        ''' None '''

        # ----------------------------------------------------------------------
        # Private Variables
        # ----------------------------------------------------------------------
      
        if os.name == 'nt':  # Skip Windows
            pass

        else:  # For Linux, macOS
            # Save the terminal settings
            self.__file_descriptor  = sys.stdin.fileno()
            self.__new_terminal     = termios.tcgetattr(self.__file_descriptor)
            self.__old_terminal     = termios.tcgetattr(self.__file_descriptor)

            # New terminal setting unbuffered
            self.__new_terminal[3] = (self.__new_terminal[3] & ~termios.ICANON & ~termios.ECHO)
            termios.tcsetattr(self.__file_descriptor, termios.TCSAFLUSH, self.__new_terminal)

            # Support normal-terminal reset at exit
            atexit.register(self.set_normal_term)  

        # ----------------------------------------------------------------------
        # Internal Method Calls
        # ----------------------------------------------------------------------
        
        ''' None '''

    # ==========================================================================
    # Public Methods
    # ==========================================================================

    #####
    def reset_terminal(self):
        ''' Resets to normal terminal window.  Note: Does not work on Windows.
        '''

        if os.name == 'nt':  # Skip Windows
            pass

        else:  # For Linux, macOS
            termios.tcsetattr(self.__file_descriptor, termios.TCSAFLUSH, self.__old_terminal)


    #####
    def get_char(self):
        ''' Get the keyboard character that has been pressed.
            Note: Do not use with "get_arrow()".
        '''

        if os.name == 'nt':  # Windows
            return msvcrt.getch().decode('utf-8')

        else:  # Linux, macOS
            return sys.stdin.read(1)


    #####
    def get_arrow(self):
        ''' Get the keyboard arrow key that has been pressed.
            0 = UP, 1 = RIGHT, 2 = DOWN, 3 = LEFT
            Note: Do not use with "get_char()".
        '''

        if os.name == 'nt':  # Windows
            msvcrt.getch() # skip 0xE0
            char = msvcrt.getch()
            values = [72, 77, 80, 75]

        else:  # Linux, macOS
            char = sys.stdin.read(3)[2]
            values = [65, 67, 66, 68]

        return values.index(ord(char.decode('utf-8')))


    #####
    def is_keyboard_hit(self):
        ''' Check whether a keyboard character is pressed.
        '''
        if os.name == 'nt':  # Windows
            return msvcrt.kbhit()

        else:  # Linux, macOS
            dr,dw,de = select([sys.stdin], [], [], 0)
            return dr != []


    #####
    def is_keyboard_hit_esc(self):
        ''' Check whether the ESC key is pressed.
        '''        
        if self.is_keyboard_hit():
            key_hit = self.get_char()
            if ord(key_hit) == 27: # ESC
                return True
        else:
            return False    


    # ==========================================================================
    # Private Methods
    # ==========================================================================
    
    ''' None '''

# ==============================================================================
# Keyboard Class (End) 
# ==============================================================================



# ==============================================================================
# Program Entry Point (Start) 
# ==============================================================================

if __name__=="__main__":
    ''' This is optional '''
    
    # Testing the Keyboard Class

    keyboard = Keyboard()

    print('Try Pressing Any Keys.')
    print("ESC = Exit")

    while True:

        # print("I am Main!")

        if keyboard.is_keyboard_hit():
            c = keyboard.get_char()
            if ord(c) == 27: # ESC
                break
            print(c)

    # keyboard.reset_terminal()    

# ==============================================================================
# Program Entry Point (End) 
# ==============================================================================



################################################################################
# END OF FILE
################################################################################