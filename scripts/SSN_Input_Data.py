import pandas as pd
from SSN_Config import SSN_Data
import os


class ssn_data(object):
    """
    A class for managing SSN data and reference data
    """

    def __init__(self,
                 obs_data_path='../input_data/GNobservations_JV_V1.22.csv',
                 obs_observer_path='../input_data/GNobservers_JV_V1.22.csv',
                 font={'family': 'sans-serif',
                       'weight': 'normal',
                       'size': 21}):
        """
        Read all reference and observational and define the search parameters
        VARIABLES APPENDED TO THE OBJECT ARE SPECIFIED AT THE END

        :param obs_data_path: Location of the observational data
        :param obs_observer_path: Location of the file containing the observer's codes and names
        :param font: Font to be used while plotting
        """

        # --------------------------------------------------------------------------------------------------------------
        print('Reading Observer data...', end="", flush=True)

        self.ssn_data = SSN_Data()

        # Use relative file paths even when running script from other directory
        # dirname = os.path.dirname(__file__)
        # obs_data_path = os.path.join(dirname, obs_data_path)
        # obs_observer_path = os.path.join(dirname, obs_observer_path)

        GN_Dat = pd.read_csv(obs_data_path, quotechar='"', encoding='utf-8', header=15)

        GN_Dat['GROUPS'] = GN_Dat['GROUPS'].astype(float)

        GN_Obs = pd.read_csv(obs_observer_path, quotechar='"', encoding='utf-8')

        print('done.', flush=True)

        # Storing variables in object-----------------------------------------------------------------------------------

        # Color specification
        self.ssn_data.ClrS = (0.74, 0.00, 0.00)
        self.ssn_data.ClrN = (0.20, 0.56, 1.00)

        self.ssn_data.Clr = [(0.00, 0.00, 0.00),
                             (0.31, 0.24, 0.00),
                             (0.43, 0.16, 0.49),
                             (0.32, 0.70, 0.30),
                             (0.45, 0.70, 0.90),
                             (1.00, 0.82, 0.67)]

        self.ssn_data.font = font  # Font to be used while plotting

        self.ssn_data.GN_Dat = GN_Dat  # Observer data containing group numbers for each observer
        self.ssn_data.GN_Obs = GN_Obs  # Observer data containing observer names and codes

        print('Done reading observer data.', flush=True)
        print(' ', flush=True)