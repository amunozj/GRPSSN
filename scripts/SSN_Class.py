import pandas as pd
import datetime
from detect_peaks import detect_peaks
import numpy as np
from astropy import convolution as conv
from scipy import signal
import os.path

class ssn_cl(object):

    """
    A class for managing SSN data and reference data
    """

    def __init__(self,
                 ref_data_path='../input_data/SC_SP_RG_DB_KM_group_areas_by_day.csv',
                 silso_path='../input_data/SN_m_tot_V2.0.csv',
                 obs_data_path='../input_data/GNObservations_JV_V1.22.csv',
                 obs_observer_path='../input_data/GNObservers_JV_V1.22.csv',
                 output_path='output/testrun',
                 font={'family': 'sans-serif',
                       'weight': 'normal',
                       'size': 21}):

        """
        Read all reference and observational and define the search parameters
        VARIABLES APPENDED TO THE OBJECT ARE SPECIFIED AT THE END

        :param ref_data_path: Location of the data to be used as reference
        :param silso_path: Location of silso's sunspot series
        :param obs_data_path: Location of the observational data
        :param obs_observer_path: Location of the file containing the observer's codes and names
        :param output_path: Location of all output files
        :param font: Font to be used while plotting
        """
        # Create output folder
        if not os.path.exists(output_path):
            os.makedirs(output_path)

        ## ----------------------------------------------------------------------------------------------------
        print('Reading Reference Data...', end="", flush=True)

        REF_Dat = pd.read_csv(ref_data_path, quotechar = '"', encoding = 'ansi',header = 0)
        print('done.')

        print('Calculating ordinal day, fractional year, and number of groups...', end="")
        REF_Dat['ORDINAL'] = REF_Dat.apply(lambda x: datetime.date(x['YEAR'].astype(int),x['MONTH'].astype(int),x['DAY'].astype(int)).toordinal(),axis=1)
        REF_Dat['FRACYEAR'] = REF_Dat.apply(lambda x: x['YEAR'].astype(int)
                                                    + (  datetime.date(x['YEAR'].astype(int),x['MONTH'].astype(int),x['DAY'].astype(int)).toordinal()
                                                       - datetime.date(x['YEAR'].astype(int),1,1).toordinal() )
                                                    / (  datetime.date(x['YEAR'].astype(int)+1,1,1).toordinal()
                                                       - datetime.date(x['YEAR'].astype(int),1,1).toordinal() )
                                          ,axis=1)

        # Turning reference areas into number of groups
        REF_Grp = REF_Dat[['FRACYEAR', 'ORDINAL', 'YEAR', 'MONTH', 'DAY']].copy()
        REF_Grp['GROUPS'] = np.nansum(np.greater(REF_Dat.values[:, 3:REF_Dat.values.shape[1] - 2], 0), axis=1)
        REF_Grp['GROUPS'] = REF_Grp['GROUPS'].astype(float)
        REF_Grp.loc[np.isnan(REF_Dat['AREA1']), 'GROUPS'] = np.nan

        # Smoothing for plotting
        Gss_1D_ker = conv.Gaussian1DKernel(75)
        REF_Dat['AVGROUPS'] = conv.convolve(REF_Grp['GROUPS'].values, Gss_1D_ker)

        print('done.', flush=True)


        ## ----------------------------------------------------------------------------------------------------
        print('Reading SILSO Data...', end="", flush=True)

        SILSO_Sn = pd.read_csv(silso_path, quotechar = '"', encoding = 'ansi', header = 0)

        # Smoothing

        swin = 8 #Smoothing window in months
        window = signal.gaussian(M=swin*6, std=swin)
        window /= window.sum()

        SILSO_Sn['MSMOOTH'] = np.convolve(SILSO_Sn['MMEAN'], window, mode='same')

        # Finding maxima and minima
        pkMax = detect_peaks(SILSO_Sn['MSMOOTH'], mpd=5)
        pkMin = detect_peaks(-SILSO_Sn['MSMOOTH'], mpd=5)

        SIL_max = SILSO_Sn.loc[pkMax, ('MSMOOTH', 'FRACYEAR')]
        SIL_min = SILSO_Sn.loc[pkMin, ('MSMOOTH', 'FRACYEAR')]

        # Identify minima covered by the reference data
        REF_min = SIL_min.loc[np.logical_and(SIL_min['FRACYEAR'] <= np.max(REF_Dat['FRACYEAR']),
                                             SIL_min['FRACYEAR'] >= np.min(REF_Dat['FRACYEAR'])), ('MSMOOTH', 'FRACYEAR')]

        REF_max = SIL_max.loc[np.logical_and(SIL_max['FRACYEAR'] <= np.max(REF_Dat['FRACYEAR']),
                                             SIL_max['FRACYEAR'] >= np.min(REF_Dat['FRACYEAR'])), ('MSMOOTH', 'FRACYEAR')]

        print('done.', flush=True)

        # -------------------------------------------------------------------------------------------------------------
        print('Finding internal endpoints and centers of SILSO and Reference...', end="", flush=True)

        # Assinging max (1) and min (-1) labels to endpoints
        maxPointsS = np.expand_dims(SIL_max['FRACYEAR'], 1)
        maxPointsS = np.concatenate((maxPointsS, maxPointsS * 0 + 1), axis=1)

        minPointsS = np.expand_dims(SIL_min['FRACYEAR'], 1)
        minPointsS = np.concatenate((minPointsS, minPointsS * 0 - 1), axis=1)

        # Creating endpoints matrix
        endPointsS = np.append(maxPointsS, minPointsS, axis=0)

        # Sorting endpoints
        endPointsS = endPointsS[endPointsS[:, 0].argsort()]

        # Finding centers and classifying them as rising (1) and decaying (-1)
        cenPointsS = (endPointsS[1:endPointsS.shape[0], :] + endPointsS[0:endPointsS.shape[0] - 1, :]) / 2
        cenPointsS[:, 1] = endPointsS[1:endPointsS.shape[0], 1]

        # Finding internal endpoints and centers of Reference
        endPointsR = endPointsS[np.logical_and(endPointsS[:, 0] > np.min(REF_Dat['FRACYEAR']),
                                                         endPointsS[:, 0] < np.max(REF_Dat['FRACYEAR'])), :]
        cenPointsR = (endPointsR[1:endPointsR.shape[0], :] + endPointsR[0:endPointsR.shape[0] - 1, :]) / 2
        cenPointsR[:, 1] = endPointsR[1:endPointsR.shape[0], 1]

        print('done.', flush=True)


        #--------------------------------------------------------------------------------------------------------------
        print('Reading Observer data...', end="", flush=True)

        GN_Dat = pd.read_csv(obs_data_path, quotechar = '"', encoding = 'ansi',header = 15)

        GN_Dat['GROUPS'] = GN_Dat['GROUPS'].astype(float)

        GN_Obs = pd.read_csv(obs_observer_path, quotechar = '"', encoding = 'ansi')

        print('done.', flush=True)

        # Storing variables in object-----------------------------------------------------------------------------------

        # Color specification
        self.ClrS = (0.74, 0.00, 0.00)
        self.ClrN = (0.20, 0.56, 1.00)

        self.Clr = [(0.00, 0.00, 0.00),
                    (0.31, 0.24, 0.00),
                    (0.43, 0.16, 0.49),
                    (0.32, 0.70, 0.30),
                    (0.45, 0.70, 0.90),
                    (1.00, 0.82, 0.67)]

        self.output_path = output_path  # Location of all output files

        self.font = font  # Font to be used while plotting

        self.REF_Dat = REF_Dat  # Reference data with individual group areas each day
        self.GN_Dat = GN_Dat  # Observer data containing group numbers for each observer
        self.GN_Obs = GN_Obs  # Observer data containing observer names and codes

        self.endPoints = {'SILSO': endPointsS}  # Variable that stores the boundaries of each rising and decaying phase
        self.cenPoints = {'SILSO': cenPointsS}  # Variable that stores the centers of each rising and decaying phase

        print('Done initializing data.', flush=True)
        print(' ', flush=True)