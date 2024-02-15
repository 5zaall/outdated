################################################################################
# Filename          : colortable.py
# 
# Description       : This file is responsible for providing numerical values
#                     for a selected color.
#
# Created by        : Brian Vergara Rosario on 23 June 2021
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
# Foreground Class (Start) 
# ==============================================================================

class Fore:
    """ This class is responsible for providing numerical values for colors that
        are used in the foreground, e.g. text color.  

        Available Colors: BLACK, RED, GREEN, YELLOW, BLUE, MAGENTA, CYAN, 
        LIGHT_GRAY, DARK_GRAY, LIGHT_RED, LIGHT_GREEN, LIGHT_YELLOW, LIGHT_BLUE,
        LIGHT_MAGENTA, LIGHT_CYAN, and WHITE
    """
    
    BLACK           = 0
    RED             = 1
    GREEN           = 2
    YELLOW          = 3
    BLUE            = 4
    MAGENTA         = 5
    CYAN            = 6
    LIGHT_GRAY      = 7
    DARK_GRAY       = 8
    LIGHT_RED       = 9
    LIGHT_GREEN     = 10
    LIGHT_YELLOW    = 11
    LIGHT_BLUE      = 12
    LIGHT_MAGENTA   = 13
    LIGHT_CYAN      = 14
    WHITE           = 15

# ==============================================================================
# Foreground Class (End) 
# ==============================================================================


# ==============================================================================
# Background Class (Start) 
# ==============================================================================

class Back:
    """ This class is responsible for providing numerical values for colors that
        are used in the background, e.g. background of a text.

        Available Colors: BLACK, RED, GREEN, YELLOW, BLUE, MAGENTA, CYAN, 
        LIGHT_GRAY, DARK_GRAY, LIGHT_RED, LIGHT_GREEN, LIGHT_YELLOW, LIGHT_BLUE,
        LIGHT_MAGENTA, LIGHT_CYAN, and WHITE
    """
    
    BLACK           = 0
    RED             = 1
    GREEN           = 2
    YELLOW          = 3
    BLUE            = 4
    MAGENTA         = 5
    CYAN            = 6
    LIGHT_GRAY      = 7
    DARK_GRAY       = 8
    LIGHT_RED       = 9
    LIGHT_GREEN     = 10
    LIGHT_YELLOW    = 11
    LIGHT_BLUE      = 12
    LIGHT_MAGENTA   = 13
    LIGHT_CYAN      = 14
    WHITE           = 15

# ==============================================================================
# Background Class (End) 
# ==============================================================================


# ==============================================================================
# Style Class (Start) 
# ==============================================================================

class Style:
    """ This class is responsible for providing numerical values for styles that
        are used in a text.  Do note that some style might not work on different
        systems.

        Available Styles: BOLD, DIM, UNDERLINED, BLINK, REVERSE, HIDDEN,
        and RESET
    """
    
    BOLD            = 1
    DIM             = 2
    UNDERLINED      = 4
    BLINK           = 5
    REVERSE         = 7
    HIDDEN          = 8
    RESET           = 0

# ==============================================================================
# Style Class (End) 
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