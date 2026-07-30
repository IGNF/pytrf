#-------------------------------------------------------------------------------
# Copyright (c) Institut national de l'information géographique et forestière
#
# Main author:
#  - Paul Rebischung
#
# This file is part of pytrf: https://github.com/IGNF/pytrf
#
# pytrf is licensed under the MIT license found in the LICENSE.md file
# in the root directory of this source tree.
#-------------------------------------------------------------------------------



"""
pytrf configuration utilities

"""



# External imports
#-----------------
import os
from platformdirs import user_config_dir

# Internal imports
#-----------------
from pytrf.io import read_yaml, write_yaml
from pytrf.utils import record



# pytrf's configuration file
_config_file = user_config_dir('pytrf') + '/config.yaml'

# Read configuration file
def _read_config():
    if os.path.isfile(_config_file):
        return read_yaml(_config_file)
    else:
        return None
    
# Write configuration file
def _write_config(config):
    if not(os.path.isdir(user_config_dir('pytrf'))):
        os.mkdir(user_config_dir('pytrf'))
    write_yaml(config, _config_file)
    
# Get user's agency name
#-----------------------
def get_agency():
    
    """
    Get user's agency name

    Returns
    -------
    agency : str
             User's agency name

    """
    
    config = _read_config()
    
    if (config is not None):
        if hasattr(config, 'agency'):
            return config.agency
        else:
            return 'IGN'
    else:
        return 'IGN'

# Set user's agency name
#-----------------------
def set_agency(agency):

    """
    Set user's agency name

    Parameters
    ----------
    agency : str
             User's 3-character agency name

    """
    
    if not(isinstance(agency, str)):
        raise ValueError('Please provide a 3-character string as your agency name.')

    if (len(agency) != 3):
        raise ValueError('Please provide a 3-character string as your agency name.')

    config = _read_config()
    if (config is None):
        config = record()
        
    config.agency = agency
    
    _write_config(config)
