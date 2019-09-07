"""
Everything related to extracting data from raw sources.
"""
from dateutil.parser import parse
import settings
from data_handlers.data_models import DefaultDataModel


class DataExtractor:
    """
    Container class for extracting data from various sources, generally using a
    factory pattern.

    Each extraction method should return a list containing object of class
    DataModel.
    """

    @staticmethod
    def extract_data(file_name=settings.DEFAULT_DATA_FILE_LOCATION):
        """
        Loads data from a csv file to a list of DataModel objects.
        """
        with open(file_name, 'r') as file:
            data_models_to_return = []

            data_rows = file.readlines()
            for index, row in enumerate(data_rows):
                # Will simply skip the row if there's an error.
                try:
                    data_model_from_row = DataExtractor._get_data_model_from_csv_string(row)
                    data_models_to_return.append(data_model_from_row)
                except:
                    pass

            return data_models_to_return

    @staticmethod
    def _get_data_model_from_csv_string(line_from_csv_file):
        components = line_from_csv_file.split(",")
        return DefaultDataModel(
            com_id=int(components[0]),
            sample_id=components[1],
            sample_date=parse(components[2]),
            latitude=float(components[3]),
            longitude=float(components[4]),
            state=components[5],
            doy=components[6],
            cat_area_sq_km=float(components[7]),
            hydrl_cond_cat=float(components[8]),
            mean_msst=float(components[9]),
            precip8110_cat=float(components[10]),
            t_max8110_cat=float(components[11]),
            t_mean8110_cat=float(components[12]),
            t_min8110_cat=float(components[13]),
            elev_cat=float(components[14]),
            bfi_cat=float(components[15]),
            runoff_cat=float(components[16]),
            is_pteronarcys_present=int(components[17]),
            is_baetis_present=int(components[18]),
            is_caenis_present=int(components[19]),
            is_tricorythodes_present=int(components[20]),
            is_centroptilum_procloeon_present=int(components[21])
        )


