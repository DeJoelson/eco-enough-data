"""
This module contains all settings, hard_coded parameters, and configurations for
the project.
"""
import sys
import numpy as np

# HARD CODED PARAMETERS
SEED = 42

#Numpy Settings
np.set_printoptions(threshold=sys.maxsize)

# FILE LOCATIONS AND DEFAULTS
DEFAULT_MODEL_VISUALIZATION_FOLDER = "data_files/model_visualization/"
DEFAULT_LOG_FOLDER = "logs/"
GRAPH_VIZ_LOCATION = 'C:/Program Files (x86)/Graphviz2.38/bin/'

DEFAULT_DATA_FILE_LOCATION = "data_files/csv_files/FINAL_combined_NRSA_NAMC_datasets.csv"
