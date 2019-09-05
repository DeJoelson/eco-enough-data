"""
This module contains anything to do with initial representations to machine
learning techniques of navigational flight data.
"""

import numpy as np


class DataArticulationLoaders:
    """
    This class is composed of a set of static methods (following the factory
    pattern) which take in a list of DataModel objects and transforms them to
    numpy articulations appropriate for machine learning.

    No transformations should happen in this class; simply placement of existing
    numerical data from DataModel to appropriate locations in numpy structures.
    """

    @staticmethod
    def get_standard_predictors(list_of_data_models):
        result = [
            [model.latitude for model in list_of_data_models],
            [model.longitude for model in list_of_data_models],
            [model.cat_area_sq_km for model in list_of_data_models],
            [model.hydrl_cond_cat for model in list_of_data_models],
            [model.mean_msst for model in list_of_data_models],
            [model.precip8110_cat for model in list_of_data_models],
            [model.t_max8110_cat for model in list_of_data_models],
            [model.t_mean8110_cat for model in list_of_data_models],
            [model.t_min8110_cat for model in list_of_data_models],
            [model.elev_cat for model in list_of_data_models],
            [model.bfi_cat for model in list_of_data_models],
            [model.runoff_cat for model in list_of_data_models]
        ]
        result = np.array(result)
        result = np.transpose(result)
        return result

    @staticmethod
    def get_is_pteronarcys_present(list_of_data_models):
        result = [model.is_pteronarcys_present for model in list_of_data_models]
        result = np.array(result)
        return result

    @staticmethod
    def get_is_baetis_present(list_of_data_models):
        result = [model.is_baetis_present for model in list_of_data_models]
        result = np.array(result)
        return result

    @staticmethod
    def get_is_caenis_present(list_of_data_models):
        result = [model.is_caenis_present for model in list_of_data_models]
        result = np.array(result)
        return result

    @staticmethod
    def get_is_tricorythodes_present(list_of_data_models):
        result = [model.is_tricorythodes_present for model in list_of_data_models]
        result = np.array(result)
        return result

    @staticmethod
    def get_is_centroptilum_procloeon_present(list_of_data_models):
        result = [model.is_centroptilum_procloeon_present for model in list_of_data_models]
        result = np.array(result)
        return result
