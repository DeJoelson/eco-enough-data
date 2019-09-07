"""
Everything to do with the logical representation of the data happens here.
"""
import numpy as np
import matplotlib.pyplot as plt


class DefaultDataModel:
    """
    The logical data model containing all facets of our inputs and outputs.

    Data transformations done in preprocessing should happen here, and coded
    using the lazy initialization pattern via the @property decorator.
    """
    def __init__(self,
                 com_id=None,
                 sample_id=None,
                 sample_date=None,
                 latitude=None,
                 longitude=None,
                 state=None,
                 doy=None,
                 cat_area_sq_km=None,
                 hydrl_cond_cat=None,
                 mean_msst=None,
                 precip8110_cat=None,
                 t_max8110_cat=None,
                 t_mean8110_cat=None,
                 t_min8110_cat=None,
                 elev_cat=None,
                 bfi_cat=None,
                 runoff_cat=None,
                 is_pteronarcys_present=None,
                 is_baetis_present=None,
                 is_caenis_present=None,
                 is_tricorythodes_present=None,
                 is_centroptilum_procloeon_present=None
                 ):
        self.com_id = com_id
        self.sample_id = sample_id
        self.sample_date = sample_date
        self.latitude = latitude
        self.longitude = longitude
        self.state = state
        self.doy = doy
        self.cat_area_sq_km = cat_area_sq_km
        self.hydrl_cond_cat = hydrl_cond_cat
        self.mean_msst = mean_msst
        self.precip8110_cat = precip8110_cat
        self.t_max8110_cat = t_max8110_cat
        self.t_mean8110_cat = t_mean8110_cat
        self.t_min8110_cat = t_min8110_cat
        self.elev_cat = elev_cat
        self.bfi_cat = bfi_cat
        self.runoff_cat = runoff_cat
        self.is_pteronarcys_present = is_pteronarcys_present
        self.is_baetis_present = is_baetis_present
        self.is_caenis_present = is_caenis_present
        self.is_tricorythodes_present = is_tricorythodes_present
        self.is_centroptilum_procloeon_present = is_centroptilum_procloeon_present
