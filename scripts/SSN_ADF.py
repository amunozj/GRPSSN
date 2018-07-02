import pandas as pd
import datetime
import numpy as np
import sys
from astropy import convolution as conv
from scipy import signal
import scipy as sp
from copy import copy
from pyemd import emd
import os.path
from itertools import groupby

from SSN_Input_Data import ssn_data
import SSN_ADF_Plotter
from SSN_Config import SSN_ADF_Config

parent_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'functions')
sys.path.insert(1, parent_dir)  # add to pythonpath
from detect_peaks import detect_peaks



class ssnADF(ssn_data):
    """
    A class for managing SSN data, reference data, and performing ADF calculations
    """

    def __init__(self,
                 ref_data_path='input_data/SC_SP_RG_DB_KM_group_areas_by_day.csv',
                 silso_path='input_data/SN_m_tot_V2.0.csv',
                 obs_data_path='input_data/GNObservations_JV_V1.22.csv',
                 obs_observer_path='input_data/GNObservers_JV_V1.22.csv',
                 output_path='output',
                 font=None,
                 dt=30,
                 phTol=2,
                 thN=50,
                 thI=1,
                 plot=True):

        """
        Read all reference and observational and define the search parameters
        VARIABLES APPENDED TO THE OBJECT ARE SPECIFIED AT THE END

        :param ref_data_path: Location of the data to be used as reference
        :param silso_path: Location of silso's sunspot series
        :param obs_data_path: Location of the observational data
        :param obs_observer_path: Location of the file containing the observer's codes and names
        :param font: Font to be used while plotting
        :param dt: Temporal Stride in days
        :param phTol: Cycle phase tolerance in years
        :param thN: Number of thresholds including 0
        :param thI: Threshold increments
        :param plot: Flag that enables the plotting and saving of relevant figures
        """

        if font is None:
            font = {'family': 'sans-serif', 'weight': 'normal', 'size': 21}

        # Use relative file paths even when running script from other directory
        dirname = os.path.dirname(__file__)
        ref_data_path = os.path.join(dirname, ref_data_path)
        silso_path = os.path.join(dirname, silso_path)
        obs_data_path = os.path.join(dirname, obs_data_path)
        obs_observer_path = os.path.join(dirname, obs_observer_path)
        output_path = os.path.join(dirname, output_path)

        ssn_data.__init__(self, obs_data_path=obs_data_path,
                          obs_observer_path=obs_observer_path,
                          font=font)

        # Create output folder
        if not os.path.exists(output_path):
            os.makedirs(output_path)

        # ----------------------------------------------------------------------------------------------------
        print('Reading Reference Data...', end="", flush=True)

        REF_Dat = pd.read_csv(ref_data_path, quotechar='"', encoding='utf-8', header=0)
        print('done.')

        print('Calculating ordinal day, fractional year, and number of groups...', end="")
        REF_Dat['ORDINAL'] = REF_Dat.apply(
            lambda x: datetime.date(x['YEAR'].astype(int), x['MONTH'].astype(int), x['DAY'].astype(int)).toordinal(),
            axis=1)
        REF_Dat['FRACYEAR'] = REF_Dat.apply(lambda x: x['YEAR'].astype(int)
                                                      + (datetime.date(x['YEAR'].astype(int), x['MONTH'].astype(int),
                                                                       x['DAY'].astype(int)).toordinal()
                                                         - datetime.date(x['YEAR'].astype(int), 1, 1).toordinal())
                                                        / (datetime.date(x['YEAR'].astype(int) + 1, 1, 1).toordinal()
                                                           - datetime.date(x['YEAR'].astype(int), 1, 1).toordinal())
                                            , axis=1)

        # Turning reference areas into number of groups
        REF_Grp = REF_Dat[['FRACYEAR', 'ORDINAL', 'YEAR', 'MONTH', 'DAY']].copy()
        REF_Grp['GROUPS'] = np.nansum(np.greater(REF_Dat.values[:, 3:REF_Dat.values.shape[1] - 2], 0), axis=1)
        REF_Grp['GROUPS'] = REF_Grp['GROUPS'].astype(float)
        REF_Grp.loc[np.isnan(REF_Dat['AREA1']), 'GROUPS'] = np.nan

        # Smoothing for plotting
        Gss_1D_ker = conv.Gaussian1DKernel(75)
        REF_Dat['AVGROUPS'] = conv.convolve(REF_Grp['GROUPS'].values, Gss_1D_ker)

        print('done.', flush=True)

        # ----------------------------------------------------------------------------------------------------
        print('Reading SILSO Data...', end="", flush=True)

        SILSO_Sn = pd.read_csv(silso_path, quotechar='"', encoding='utf-8', header=0)

        # Smoothing
        swin = 8  # Smoothing window in months
        window = signal.gaussian(M=swin * 6, std=swin)
        window /= window.sum()

        SILSO_Sn['MSMOOTH'] = np.convolve(SILSO_Sn['MMEAN'], window, mode='same')

        # Finding maxima and minima
        pkMax = detect_peaks(SILSO_Sn['MSMOOTH'], mpd=5)
        pkMin = detect_peaks(-SILSO_Sn['MSMOOTH'], mpd=5)

        SIL_max = SILSO_Sn.loc[pkMax, ('MSMOOTH', 'FRACYEAR')]
        SIL_min = SILSO_Sn.loc[pkMin, ('MSMOOTH', 'FRACYEAR')]

        # Identify minima covered by the reference data
        REF_min = SIL_min.loc[np.logical_and(SIL_min['FRACYEAR'] <= np.max(REF_Dat['FRACYEAR']),
                                             SIL_min['FRACYEAR'] >= np.min(REF_Dat['FRACYEAR'])), (
                                  'MSMOOTH', 'FRACYEAR')]

        REF_max = SIL_max.loc[np.logical_and(SIL_max['FRACYEAR'] <= np.max(REF_Dat['FRACYEAR']),
                                             SIL_max['FRACYEAR'] >= np.min(REF_Dat['FRACYEAR'])), (
                                  'MSMOOTH', 'FRACYEAR')]

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

        # --------------------------------------------------------------------------------------------------
        print('Creating window masks...', end="", flush=True)

        risMask = {'MASK': np.zeros(REF_Dat.shape[0], dtype=bool)}
        decMask = {'MASK': np.zeros(REF_Dat.shape[0], dtype=bool)}

        # Applying mask
        for cen in cenPointsR:
            if cen[1] == 1:
                risMask['MASK'][np.logical_and(REF_Dat['FRACYEAR'].values >= cen[0] - phTol,
                                               REF_Dat['FRACYEAR'].values <= cen[0] + phTol)] = True
            else:
                decMask['MASK'][np.logical_and(REF_Dat['FRACYEAR'].values >= cen[0] - phTol,
                                               REF_Dat['FRACYEAR'].values <= cen[0] + phTol)] = True

        # Creating cadence mask
        cadMask = np.zeros(REF_Dat.shape[0], dtype=bool)
        cadMask[range(0, cadMask.shape[0], dt)] = True

        # Storing maks for plotting
        risMask['PLOT'] = risMask['MASK']
        decMask['PLOT'] = decMask['MASK']

        # Creating rising mask
        risMask['MASK'] = np.logical_and(cadMask, risMask['MASK'])

        # Creating decaying mask
        decMask['MASK'] = np.logical_and(cadMask, decMask['MASK'])

        # Turnings Mask into indices
        risMask['INDEX'] = np.array(risMask['MASK'].nonzero()[0])
        decMask['INDEX'] = np.array(decMask['MASK'].nonzero()[0])

        print('done.', flush=True)

        # Storing variables in object-----------------------------------------------------------------------------------

        self.ssn_data.output_path = output_path  # Location of all output files

        self.ssn_data.font = font  # Font to be used while plotting
        self.ssn_data.ssn_datadt = dt  # Temporal Stride in days
        self.ssn_data.phTol = phTol  # Cycle phase tolerance in years
        self.ssn_data.thN = thN  # Number of thresholds including 0
        self.ssn_data.thI = thI  # Threshold increments

        self.ssn_data.REF_Dat = REF_Dat  # Reference data with individual group areas each day

        self.ssn_data.risMask = risMask  # Mask indicating where to place the search window during raising phases
        self.ssn_data.decMask = decMask  # Mask indicating where to place the search window during declining phases

        self.ssn_data.endPoints = {
            'SILSO': endPointsS}  # Variable that stores the boundaries of each rising and decaying phase
        self.ssn_data.cenPoints = {
            'SILSO': cenPointsS}  # Variable that stores the centers of each rising and decaying phase

        # --------------------------------------------------------------------------------------------------------------

        if plot:
            SSN_ADF_Plotter.plotSearchWindows(self.ssn_data, SILSO_Sn, SIL_max, SIL_min, REF_min, REF_max)

        print('Done initializing data.', flush=True)
        print(' ', flush=True)

    def processObserver(self,
                        ssn_data,
                        CalObs=412,
                        MoLngt=30,
                        minObD=0.33,
                        vldIntThr=0.33):

        """
        Function that breaks a given observer's data into "months", calculates the ADF and breaks it into rising and
        decaying intervals
        VARIABLES APPENDED TO THE OBJECT ARE SPECIFIED AT THE END

        :param CalObs: Observer identifier denoting observer to be processed
        :param MoLngt: Duration of the interval ("month") used to calculate the ADF
        :param minObD: Minimum proportion of days with observation for a "month" to be considered valid
        :param vldIntThr: Minimum proportion of valid "months" for a decaying or raising interval to be considered valid
        :return:  (False) True if there are (no) valid intervals
        """

        NamObs = ssn_data.GN_Obs['OBSERVER'].values[ssn_data.GN_Obs['STATION'].values == CalObs]
        NamObs = NamObs[0]
        NamObs = NamObs[0:NamObs.find(',')].capitalize()

        print('Processing ' + NamObs, flush=True)

        # Picking observations
        ObsDat = ssn_data.GN_Dat[ssn_data.GN_Dat.STATION == CalObs].copy()

        # If no data for observer exit
        if ObsDat.shape[0] == 0:
            print('done. NO VALID INTERVALS IN OBSERVER', flush=True)
            print(' ', flush=True)
            ssn_data.CalObs = CalObs
            ssn_data.NamObs = NamObs
            ssn_data.minObD = minObD
            ssn_data.MoLngt = MoLngt
            ssn_data.ObsDat = ObsDat
            return False

        # Finding missing days
        ObsInt = np.arange(np.min(ObsDat['ORDINAL']), np.max(ObsDat['ORDINAL'] + 1))
        MisDays = np.logical_not(sp.in1d(ObsInt, ObsDat['ORDINAL']))

        # Creating dataframe with NaNs for days without observations
        year = np.array(list(map(lambda x: datetime.date.fromordinal(x).year, ObsInt[MisDays])))
        month = np.array(list(map(lambda x: datetime.date.fromordinal(x).month, ObsInt[MisDays])))
        day = np.array(list(map(lambda x: datetime.date.fromordinal(x).day, ObsInt[MisDays])))

        station = day * 0 + CalObs
        observer = day * 0 + 1
        groups = day * np.nan

        fractyear = np.array(list(map(lambda year, month, day: year + (datetime.date(year, month, day).toordinal()
                                                                       - datetime.date(year, 1, 1).toordinal())
                                                                      / (datetime.date(year + 1, 1, 1).toordinal()
                                                                         - datetime.date(year, 1, 1).toordinal()),
                                      year, month, day)))

        NoObs = pd.DataFrame(np.column_stack((year, month, day, ObsInt[MisDays], station, observer, groups, fractyear)),
                             columns=ObsDat.columns.values)

        # Append dataframe with missing days
        ObsDat = ObsDat.append(NoObs, ignore_index=True)

        # Recast using original data types
        origType = ssn_data.GN_Dat.dtypes.to_dict()
        ObsDat = ObsDat.apply(lambda x: x.astype(origType[x.name]))

        # Sorting according to date
        ObsDat = ObsDat.sort_values('ORDINAL').reset_index(drop=True)

        print('Calculating variables for plotting observer...', flush=True)

        # Selecting the maximum integer amount of "months" out of the original data
        yrOb = ObsDat['FRACYEAR'].values
        yrOb = yrOb[0:np.int(yrOb.shape[0] / MoLngt) * MoLngt]

        grpsOb = ObsDat['GROUPS'].values
        grpsOb = grpsOb[0:np.int(grpsOb.shape[0] / MoLngt) * MoLngt]

        # Reshaping
        yrOb = yrOb.reshape((-1, MoLngt))

        # If no data for observer exit
        if yrOb.shape[0] == 0:
            print('done. NO VALID MONTHS IN OBSERVER', flush=True)
            print(' ', flush=True)
            ssn_data.CalObs = CalObs
            ssn_data.NamObs = NamObs
            ssn_data.minObD = minObD
            ssn_data.MoLngt = MoLngt
            ssn_data.ObsDat = ObsDat
            return False

        grpsOb = grpsOb.reshape((-1, MoLngt))

        # Interval edges for plotting
        fyr1Ob = np.min(yrOb, axis=1)
        fyr2Ob = np.max(yrOb, axis=1)

        obsPlt = {'X': (fyr1Ob + fyr2Ob) / 2}

        # Average number of groups
        Gss_1D_ker = conv.Gaussian1DKernel(2)
        obsPlt['Y'] = conv.convolve(np.nanmean(grpsOb, axis=1), Gss_1D_ker)

        # Finding internal endpoints and centers of Observer Intervals are included if their center is covered by the observer

        # Defining boolean array of valid centers
        validCen = np.logical_and(ssn_data.cenPoints['SILSO'][:, 0] > np.min(yrOb),
                                  ssn_data.cenPoints['SILSO'][:, 0] < np.max(yrOb))

        # Adding a True on the index prior to the first center to include the bracketing point
        validCen[0:validCen.shape[0] - 2] = np.logical_or(validCen[0:validCen.shape[0] - 2],
                                                          validCen[1:validCen.shape[0] - 1])

        # Adding a False at the beggining to account for the difference in size
        validCen = np.insert(validCen, 0, False)

        # Defining arrays
        endPoints = ssn_data.endPoints['SILSO'][validCen, :]

        if endPoints.shape[0] == 0:
            endPoints = ssn_data.endPoints['SILSO'][0:2, :]
            endPoints[0, 0] = np.min(yrOb)
            endPoints[1, 0] = np.max(yrOb)

        cenPoints = (endPoints[1:endPoints.shape[0], :] + endPoints[0:endPoints.shape[0] - 1, :]) / 2
        cenPoints[:, 1] = endPoints[1:endPoints.shape[0], 1]

        # Identification of Min-Max Max-Min intervals with enough valid "months"
        vldIntr = np.zeros(cenPoints.shape[0], dtype=bool)

        for siInx in range(0, cenPoints.shape[0]):

            # Redefining endpoints if interval is partial
            if endPoints[siInx, 0] < np.min(ObsDat['FRACYEAR']):
                print('Redefining left endpoint')
                endPoints[siInx, 0] = np.min(ObsDat['FRACYEAR'])
                cenPoints[siInx, 0] = (endPoints[siInx, 0] + endPoints[siInx + 1, 0]) / 2

            if endPoints[siInx + 1, 0] > np.max(ObsDat['FRACYEAR']):
                print('Redefining right endpoint')
                endPoints[siInx + 1, 0] = np.max(ObsDat['FRACYEAR'])
                cenPoints[siInx, 0] = (endPoints[siInx, 0] + endPoints[siInx + 1, 0]) / 2

            print('Center:', np.round(cenPoints[siInx, 0], 2), 'Edges:', np.round(endPoints[siInx, 0], 2),
                  np.round(endPoints[siInx + 1, 0], 2))

            # Selecting interval
            TObsDat = ObsDat.loc[np.logical_and(ObsDat['FRACYEAR'] >= endPoints[siInx, 0],
                                                ObsDat['FRACYEAR'] < endPoints[siInx + 1, 0]), 'GROUPS'].values.copy()

            # Selecting the maximum integer amount of "months" out of the original data
            TgrpsOb = TObsDat[0:np.int(TObsDat.shape[0] / MoLngt) * MoLngt].copy()

            # Reshaping into "months"
            TgrpsOb = TgrpsOb.reshape((-1, MoLngt))


            # Number of days with observations
            ODObs = np.sum(np.isfinite(TgrpsOb), axis=1)

            if np.sum(ODObs / MoLngt >= minObD) / ODObs.shape[0] >= vldIntThr:
                # Marking interval as valid
                vldIntr[siInx] = True
                print('Valid interval. Proportion of valid months: ',
                      np.round(np.sum(ODObs / MoLngt >= minObD) / ODObs.shape[0], 2))

            else:
                print('INVALID interval. Proportion of valid months: ',
                      np.round(np.sum(ODObs / MoLngt >= minObD) / ODObs.shape[0], 2))

            print(' ')

        print(str(np.sum(vldIntr)) + '/' + str(vldIntr.shape[0]) + ' valid intervals')

        # Find valid months
        vMonths = []
        for m in grpsOb:
            l = len(m)
            m = [0 if np.isnan(y) else 1 for y in m]
            vMonths.append(True if (np.sum(m) / l > minObD) else False)

        # Storing variables in object-----------------------------------------------------------------------------------

        ssn_data.CalObs = CalObs  # Observer identifier denoting observer to be processed
        ssn_data.NamObs = NamObs  # Name of observer
        ssn_data.minObD = minObD  # Minimum fraction of observed days for an interval to be considered useful
        ssn_data.MoLngt = MoLngt  # Duration of the interval ("month") used to calculate the ADF
        ssn_data.ObsDat = ObsDat  # Data of observer being analyzed

        ssn_data.endPoints['OBS'] = endPoints  # Variable that stores the boundaries of each rising and decaying phase
        ssn_data.cenPoints['OBS'] = cenPoints  # Variable that stores the centers of each rising and decaying phase

        ssn_data.vldIntr = vldIntr  # Stores whether each rising or decaying interval has enough data to be valid
        ssn_data.obsPlt = obsPlt  # Variable with the observer average groups for plotting

        # NEW MACHINE LEARNING PARAMETERS
        ssn_data.NumMonths = yrOb.shape[0]  # NUMBER OF MONTHS OBSERVED
        ssn_data.ODobs = ODObs  # NUMBER OF DAYS WITH OBSERVATIONS

        n_qui = ObsDat.loc[ObsDat.GROUPS == 0.0].shape[0]
        n_act = np.isfinite(ObsDat.loc[ObsDat.GROUPS > 0.0]).shape[0]
        n_na = ObsDat.loc[np.isnan(ObsDat.GROUPS)].shape[0]
        n_tot = ObsDat.shape[0]
        ssn_data.QDays = n_qui  # Total number of Quiet days
        ssn_data.ADays = n_act  # Total number of Active days
        ssn_data.NADays = n_na  # Total unobserved days in data
        ssn_data.TDays = n_tot  # Total days accounted for in data
        ssn_data.QAFrac = n_qui / n_act  # Fraction of quiet to active days

        ssn_data.ObsPerMonth = (n_act + n_qui) / yrOb.shape[0]  # Average number of days observed per month

        ssn_data.RiseCount = len([x for x in cenPoints if x[1] == 1.0])
        ssn_data.DecCount = len([x for x in cenPoints if x[1] == -1.0])

        ssn_data.InvInts = np.sum(np.logical_not(vldIntr))  # Number of invalid intervals in observer

        ssn_data.InvMonths = np.sum(np.logical_not(vMonths))
        strks = [sum(1 for _ in g) for k, g in groupby(vMonths) if not k]
        if strks:
            ssn_data.InvMoStreak = max(strks)
        else:
            ssn_data.InvMoStreak = 0

        ssn_data.ObsStartDate = ObsDat['ORDINAL'].data[0]
        ssn_data.ObsTotLength = ObsDat['ORDINAL'].data[-1] - ObsDat['ORDINAL'].data[0]

        self.ssn_data = ssn_data

        # --------------------------------------------------------------------------------------------------------------

        # Create folder for observer's output
        if not os.path.exists(ssn_data.output_path + '/' + str(CalObs) + '_' + NamObs):
            os.makedirs(ssn_data.output_path + '/' + str(CalObs) + '_' + NamObs)

        if np.sum(vldIntr) == 0:
            print('done. NO VALID INTERVALS IN OBSERVER', flush=True)
            print(' ', flush=True)
            return False
        else:
            print('done. Observer has valid intervals', flush=True)
            print(' ', flush=True)

            return True

    def ADFscanningWindowEMD(self,
                             ssn_data,
                             nBest=50):

        """
        Function that preps the search windows and calculates the EMD for each separate rising and decaying interval
        comparing the observer and the reference
        VARIABLES APPENDED TO THE OBJECT ARE SPECIFIED AT THE END

        :param ssn_data: SSN_data class object storing SSN metadata
        :param nBest: Number of top best matches to keep
        :return:  (False) True if there are (no) valid days of overlap between observer and reference
        """

        print('Calculating number of active and observed days using scanning windows...', flush=True)

        # Creating Storing dictionaries
        # Number of days with groups
        GDObsI = []
        GDREFI = []

        # Number of days with observations
        ODObsI = []
        ODREFI = []

        # Number of days with no groups
        QDObsI = []
        QDREFI = []

        # Going through different sub-intervals
        num_intervals = ssn_data.cenPoints['OBS'].shape[0]
        for siInx in range(0, num_intervals):

            print('Center:', np.round(ssn_data.cenPoints['OBS'][siInx, 0], 2), 'Edges:',
                  np.round(ssn_data.endPoints['OBS'][siInx, 0], 2),
                  np.round(ssn_data.endPoints['OBS'][siInx + 1, 0], 2))

            # Perform analysis Only if the period is valid
            if ssn_data.vldIntr[siInx]:

                print('[{}/{}] Valid Interval'.format(siInx + 1, num_intervals))

                # Defining mask based on the interval type (rise or decay)
                if ssn_data.cenPoints['OBS'][siInx, 1] > 0:
                    cadMaskI = ssn_data.risMask['INDEX']
                else:
                    cadMaskI = ssn_data.decMask['INDEX']

                # Selecting interval
                TObsDat = ssn_data.ObsDat.loc[
                    np.logical_and(ssn_data.ObsDat['FRACYEAR'] >= ssn_data.endPoints['OBS'][siInx, 0],
                                   ssn_data.ObsDat['FRACYEAR'] < ssn_data.endPoints['OBS'][siInx + 1, 0])
                    , 'GROUPS'].values.copy()

                TObsFYr = ssn_data.ObsDat.loc[
                    np.logical_and(ssn_data.ObsDat['FRACYEAR'] >= ssn_data.endPoints['OBS'][siInx, 0],
                                   ssn_data.ObsDat['FRACYEAR'] < ssn_data.endPoints['OBS'][siInx + 1, 0])
                    , 'FRACYEAR'].values.copy()

                # Find index of center of sub-interval
                minYear = np.min(np.absolute(TObsFYr - ssn_data.cenPoints['OBS'][siInx, 0]))
                obsMinInx = (np.absolute(TObsFYr - ssn_data.cenPoints['OBS'][siInx, 0]) == minYear).nonzero()[0][0]

                # Creating Storing Variables
                # Number of days with groups
                GDObs = np.zeros((ssn_data.thN, cadMaskI.shape[0], np.int(TObsDat.shape[0] / ssn_data.MoLngt)))
                GDREF = np.zeros((ssn_data.thN, cadMaskI.shape[0], np.int(TObsDat.shape[0] / ssn_data.MoLngt)))

                # Number of days with observations
                ODObs = np.zeros((ssn_data.thN, cadMaskI.shape[0], np.int(TObsDat.shape[0] / ssn_data.MoLngt)))
                ODREF = np.zeros((ssn_data.thN, cadMaskI.shape[0], np.int(TObsDat.shape[0] / ssn_data.MoLngt)))

                # Number of days with no groups
                QDObs = np.zeros((ssn_data.thN, cadMaskI.shape[0], np.int(TObsDat.shape[0] / ssn_data.MoLngt)))
                QDREF = np.zeros((ssn_data.thN, cadMaskI.shape[0], np.int(TObsDat.shape[0] / ssn_data.MoLngt)))

                # Going through different thresholds
                for TIdx in range(0, ssn_data.thN):

                    # Calculating number of groups in reference data for given threshold
                    grpsREFw = np.nansum(
                        np.greater(ssn_data.REF_Dat.values[:, 3:ssn_data.REF_Dat.values.shape[1] - 3],
                                   TIdx * ssn_data.thI),
                        axis=1).astype(float)
                    grpsREFw[np.isnan(ssn_data.REF_Dat['AREA1'])] = np.nan

                    # Going through different shifts
                    for SIdx in range(0, cadMaskI.shape[0]):

                        # Selecting the maximum integer amount of "months" out of the original data
                        TgrpsOb = TObsDat[0:np.int(TObsDat.shape[0] / ssn_data.MoLngt) * ssn_data.MoLngt].copy()

                        # Calculating bracketing indices
                        Idx1 = cadMaskI[SIdx] - obsMinInx
                        Idx2 = Idx1 + TgrpsOb.shape[0]

                        # Selecting reference window of matching size to observer sub-interval;
                        TgrpsREF = grpsREFw[Idx1:Idx2].copy()

                        # Making sure selections have the same length
                        if TgrpsREF.shape[0] == TgrpsOb.shape[0]:
                            # Reshaping into "months"
                            TgrpsOb = TgrpsOb.reshape((-1, ssn_data.MoLngt))
                            TgrpsREF = TgrpsREF.reshape((-1, ssn_data.MoLngt))

                            # Imprinting missing days
                            # OBSERVER
                            TgrpsOb[np.isnan(TgrpsREF)] = np.nan
                            # REFERENCE
                            TgrpsREF[np.isnan(TgrpsOb)] = np.nan

                            # Number of days with groups
                            # OBSERVER
                            GDObs[TIdx, SIdx, :] = np.sum(np.greater(TgrpsOb, 0), axis=1)
                            # REFERENCE
                            GDREF[TIdx, SIdx, :] = np.sum(np.greater(TgrpsREF, 0), axis=1)

                            # Number of days with observations
                            # OBSERVER
                            ODObs[TIdx, SIdx, :] = np.sum(np.isfinite(TgrpsOb), axis=1)
                            # REFERENCE
                            ODREF[TIdx, SIdx, :] = np.sum(np.isfinite(TgrpsREF), axis=1)

                            # Number of days with no groups
                            # OBSERVER
                            QDObs[TIdx, SIdx, :] = np.sum(np.equal(TgrpsOb, 0), axis=1)
                            # REFERENCE
                            QDREF[TIdx, SIdx, :] = np.sum(np.equal(TgrpsREF, 0), axis=1)

            # If period is not valid append empty variavbles
            else:
                print('[{}/{}] INVALID Interval'.format(siInx + 1, num_intervals))
                GDObs = []
                GDREF = []
                ODObs = []
                ODREF = []
                QDObs = []
                QDREF = []

            print(' ')

            # Appending calculated days to list of sub-intervals
            GDObsI.append(GDObs)
            GDREFI.append(GDREF)
            ODObsI.append(ODObs)
            ODREFI.append(ODREF)
            QDObsI.append(QDObs)
            QDREFI.append(QDREF)

        print('done.', flush=True)
        print(' ', flush=True)

        print('Calculating the Earths Mover Distance using a sliding window...', flush=True)

        # Creating Storing dictionaries for distance matrices
        EMDD = []
        EMDtD = []
        EMDthD = []

        # Calculation of distance matrix to be used in the Earth Movers Metric
        x = np.arange(0, ssn_data.MoLngt + 1)
        y = np.arange(0, ssn_data.MoLngt + 1)
        xx, yy = np.meshgrid(x, y)
        Dis = np.absolute(np.power(xx - yy, 1))

        # Going through different sub-intervals
        for siInx in range(0, ssn_data.cenPoints['OBS'].shape[0]):

            print('Center:', np.round(ssn_data.cenPoints['OBS'][siInx, 0], 2), 'Edges:',
                  np.round(ssn_data.endPoints['OBS'][siInx, 0], 2),
                  np.round(ssn_data.endPoints['OBS'][siInx + 1, 0], 2))

            # Perform analysis Only if the period is valid
            if ssn_data.vldIntr[siInx]:

                print('[{}/{}] Valid Interval'.format(siInx + 1, num_intervals))

                # Defining mask based on the interval type (rise or decay)
                if ssn_data.cenPoints['OBS'][siInx, 1] > 0:
                    cadMaskI = ssn_data.risMask['INDEX']
                else:
                    cadMaskI = ssn_data.decMask['INDEX']

                # Pre-allocating EMD matrix and associated coordinate matrices.  A large default distance valued is used
                # to account for missing points
                EMD = np.ones((GDREFI[siInx].shape[0], GDREFI[siInx].shape[1])) * 1e16
                EMDt = np.zeros((GDREFI[siInx].shape[0], GDREFI[siInx].shape[1]))
                EMDth = np.zeros((GDREFI[siInx].shape[0], GDREFI[siInx].shape[1]))

                # Going through different thresholds
                for TIdx in range(0, ssn_data.thN):

                    # Going through different shifts
                    for SIdx in range(0, cadMaskI.shape[0]):

                        if np.any(ODObsI[siInx][TIdx, SIdx, :] != 0) and np.any(ODREFI[siInx][TIdx, SIdx, :] != 0):
                            # Calculating Earth Mover's Distance

                            # Change denom depending on
                            if SSN_ADF_Config.MONTH_TYPE == "FULLM":
                                ADF_Obs_denom = ssn_data.MoLngt
                                ADF_REF_denom = ssn_data.MoLngt
                            elif SSN_ADF_Config.MONTH_TYPE == "OBS":
                                ADF_Obs_denom = ODObsI[siInx][
                                    TIdx, SIdx, ODObsI[siInx][TIdx, SIdx, :] / ssn_data.MoLngt >= ssn_data.minObD]
                                ADF_REF_denom = ODREFI[siInx][
                                    TIdx, SIdx, ODREFI[siInx][TIdx, SIdx, :] / ssn_data.MoLngt >= ssn_data.minObD]
                            else:
                                raise ValueError(
                                    'Invalid flag: Use \'OBS\' (or \'FULLM\') to use observed days (full month length) to determine ADF.')

                            # Change calculation depending on ADF/QDF flag
                            if SSN_ADF_Config.ADF_TYPE == "QDF":
                                ADF_Obs_numerator = QDObsI[siInx][
                                    TIdx, SIdx, ODObsI[siInx][TIdx, SIdx, :] / ssn_data.MoLngt >= ssn_data.minObD]
                                ADF_REF_numerator = QDREFI[siInx][
                                    TIdx, SIdx, ODREFI[siInx][TIdx, SIdx, :] / ssn_data.MoLngt >= ssn_data.minObD]

                                ADF_Obs_frac = 1.0 - np.divide(ADF_Obs_numerator, ADF_Obs_denom)
                                ADF_REF_frac = 1.0 - np.divide(ADF_REF_numerator, ADF_REF_denom)
                            elif SSN_ADF_Config.ADF_TYPE == "ADF":
                                ADF_Obs_numerator = GDObsI[siInx][
                                    TIdx, SIdx, ODObsI[siInx][TIdx, SIdx, :] / ssn_data.MoLngt >= ssn_data.minObD]
                                ADF_REF_numerator = GDREFI[siInx][
                                    TIdx, SIdx, ODREFI[siInx][TIdx, SIdx, :] / ssn_data.MoLngt >= ssn_data.minObD]
                                ADF_Obs_frac = np.divide(ADF_Obs_numerator, ADF_Obs_denom)
                                ADF_REF_frac = np.divide(ADF_REF_numerator, ADF_REF_denom)
                            else:
                                raise ValueError(
                                    'Invalid flag: Use \'ADF\' (or \'QDF\') for active (1-quiet) day fraction.')

                            # Main ADF calculations
                            ADFObs, bins = np.histogram(ADF_Obs_frac,
                                                        bins=(
                                                                 np.arange(0,
                                                                           ssn_data.MoLngt + 2) - 0.5) / ssn_data.MoLngt,
                                                        density=True)

                            ADFREF, bins = np.histogram(ADF_REF_frac,
                                                        bins=(
                                                                 np.arange(0,
                                                                           ssn_data.MoLngt + 2) - 0.5) / ssn_data.MoLngt,
                                                        density=True)

                            EMD[TIdx, SIdx] = emd(ADFREF.astype(np.float64), ADFObs.astype(np.float64),
                                                  Dis.astype(np.float64))

                        # #Calculating Chi-Square distance
                        #                 ADFObs, bins = np.histogram(GDObsI[siInx][TIdx,SIdx,ODObsI[siInx][TIdx,SIdx,:]/MoLngt>=minObD]/MoLngt, bins= (np.arange(0,MoLngt+2)-0.5)/MoLngt)
                        #                 ADFREF, bins = np.histogram(GDREFI[siInx][TIdx,SIdx,ODREFI[siInx][TIdx,SIdx,:]/MoLngt>=minObD]/MoLngt, bins= (np.arange(0,MoLngt+2)-0.5)/MoLngt)

                        #                 # Calculating numerator and denominator for Chi-square distance
                        #                 Nom = np.power(ADFObs-ADFREF,2)
                        #                 #Den = np.power(ADFObs,2) + np.power(ADFREF,2)
                        #                 Den = ADFObs + ADFREF

                        #                 # Removing zeros in denominator
                        #                 Nom = Nom[Den!=0]
                        #                 Den = Den[Den!=0]

                        #                 # Calculating Chi-square distance
                        #                 EMD[TIdx,SIdx] = np.sum(np.divide(Nom,Den))

                        # Storing coordinates of EMD distances
                        EMDt[TIdx, SIdx] = ssn_data.REF_Dat['FRACYEAR'].values[cadMaskI[SIdx]]
                        EMDth[TIdx, SIdx] = TIdx * ssn_data.thI

            # If period is not valid append empty variables
            else:
                print('[{}/{}] INVALID Interval'.format(siInx + 1, num_intervals))
                EMD = []
                EMDt = []
                EMDth = []

            print(' ')

            EMDD.append(EMD)
            EMDtD.append(EMDt)
            EMDthD.append(EMDth)

        print('done.', flush=True)
        print(' ', flush=True)

        print('Identifying the best matches for each valid period and looking for ref-obs overlap...', end="",
              flush=True)

        # Creating Storing dictionaries to store best thresholds
        bestTh = []
        calRef = []
        calObs = []

        # Creating storing dictionaries to store fit properties
        rSqI = []
        mResI = []

        # Switch indicating that there is overlap between reference and observer
        obsRefOvrlp = False

        # Variables to store the mean threshold and its standard deviation
        wAvI = ssn_data.vldIntr.copy() * 0
        wSDI = wAvI.copy()

        for siInx in range(0, ssn_data.cenPoints['OBS'].shape[0]):

            # Analyze period only if valid
            if ssn_data.vldIntr[siInx]:

                # Creating matrix for sorting and find the best combinations of threshold and shift
                OpMat = np.concatenate(
                    (EMDtD[siInx].reshape((-1, 1)), EMDthD[siInx].reshape((-1, 1)),
                     EMDD[siInx].reshape((-1, 1))),
                    axis=1)

                # Sort according to EMD to find the best matches
                I = np.argsort(OpMat[:, 2], axis=0)
                OpMat = np.squeeze(OpMat[I, :])

                # Adding best points
                bestTh.append(OpMat[0:nBest - 1, :])

                # Constructing weights
                alph = 1 - (bestTh[siInx][:, 2] - np.min(bestTh[siInx][:, 2])) / (
                    np.max(bestTh[siInx][:, 2]) - np.min(bestTh[siInx][:, 2]))

                if np.isnan(np.sum(alph)):
                    alph = bestTh[siInx][:, 2] * 0 + 1

                # Weighted average
                wAvI[siInx] = np.sum(np.multiply(alph, bestTh[siInx][:, 1])) / np.sum(alph)

                # Weighted Standard Deviation
                wSDI[siInx] = np.sqrt(
                    np.sum(np.multiply(alph, np.power(bestTh[siInx][:, 1] - wAvI[siInx], 2))) / np.sum(alph))

                if np.sum(np.logical_and(ssn_data.REF_Dat['FRACYEAR'] >= ssn_data.endPoints['OBS'][siInx, 0],
                                         ssn_data.REF_Dat['FRACYEAR'] < ssn_data.endPoints['OBS'][siInx + 1, 0])) > 0:

                    # Activate the overlap switch
                    obsRefOvrlp = True

                    # Calculating number of groups in reference data for given threshold
                    grpsREFw = np.nansum(
                        np.greater(ssn_data.REF_Dat.values[:, 3:ssn_data.REF_Dat.values.shape[1] - 3], wAvI[siInx]),
                        axis=1).astype(float)
                    grpsREFw[np.isnan(ssn_data.REF_Dat['AREA1'])] = np.nan

                    # Selecting observer's interval
                    TObsDat = ssn_data.ObsDat.loc[
                        np.logical_and(ssn_data.ObsDat['FRACYEAR'] >= ssn_data.endPoints['OBS'][siInx, 0],
                                       ssn_data.ObsDat['FRACYEAR'] < ssn_data.endPoints['OBS'][siInx + 1, 0])
                        , 'GROUPS'].values.copy()
                    TObsOrd = ssn_data.ObsDat.loc[
                        np.logical_and(ssn_data.ObsDat['FRACYEAR'] >= ssn_data.endPoints['OBS'][siInx, 0],
                                       ssn_data.ObsDat['FRACYEAR'] < ssn_data.endPoints['OBS'][siInx + 1, 0])
                        , 'ORDINAL'].values.copy()

                    # Selecting the days of overlap with calibrated observer
                    grpsREFw = grpsREFw[np.in1d(ssn_data.REF_Dat['ORDINAL'].values, TObsOrd)]
                    grpsObsw = TObsDat[np.in1d(TObsOrd, ssn_data.REF_Dat['ORDINAL'].values)]

                    # Removing NaNs
                    grpsREFw = grpsREFw[np.isfinite(grpsObsw)]
                    grpsObsw = grpsObsw[np.isfinite(grpsObsw)]

                    grpsObsw = grpsObsw[np.isfinite(grpsREFw)]
                    grpsREFw = grpsREFw[np.isfinite(grpsREFw)]

                    # Calculating goodness of fit of Y=X

                    # R squared
                    yMean = np.mean(grpsREFw)
                    SStot = np.sum(np.power(grpsREFw - yMean, 2))
                    SSreg = np.sum(np.power(grpsREFw - grpsObsw, 2))
                    rSq = 1 - SSreg / SStot

                    # Mean Residual
                    mRes = np.mean(grpsREFw - grpsObsw)

                    calRef.append(grpsREFw)
                    calObs.append(grpsObsw)
                    rSqI.append(rSq)
                    mResI.append(mRes)

                else:
                    calRef.append([])
                    calObs.append([])
                    rSqI.append([])
                    mResI.append([])

            # If period not valid store an empty array
            else:
                bestTh.append([])
                calRef.append([])
                calObs.append([])
                rSqI.append([])
                mResI.append([])

        rSqDT = np.nan
        mResDT = np.nan

        # Only if there is at leas one interval that is valid
        if len(calRef) > 0:
            # Calculating r square and mean residual using different thresholds for different intervals
            tcalRef = np.concatenate(calRef, axis=0)
            tcalObs = np.concatenate(calObs, axis=0)

            # R squared
            yMean = np.mean(tcalRef)
            SStot = np.sum(np.power(tcalRef - yMean, 2))
            SSreg = np.sum(np.power(tcalRef - tcalObs, 2))
            rSqDT = 1 - SSreg / SStot

            # Mean Residual
            mResDT = np.mean(tcalRef - tcalObs)

        # Storing variables in object-----------------------------------------------------------------------------------

        ssn_data.GDObsI = GDObsI  # Variable that stores the number of days with groups of the observer for each interval, threshold, window shift, and window
        ssn_data.ODObsI = ODObsI  # Variable that stores the number of days with observations of the observer for each interval, threshold, window shift, and window
        ssn_data.GDREFI = GDREFI  # Variable that stores the number of days with groups of the reference for each interval, threshold, window shift, and window
        ssn_data.ODREFI = ODREFI  # Variable that stores the number of days with observations of the reference for each interval, threshold, window shift, and window

        ssn_data.EMDD = EMDD  # Variable that stores the EMD between the reference and the observer for each interval, threshold, and window shift
        ssn_data.EMDtD = EMDtD  # Variable that stores the windowshift matching EMDD for each interval, threshold, and window shift
        ssn_data.EMDthD = EMDthD  # Variable that stores the threshold matching EMDD for each interval, threshold, and window shift

        ssn_data.Dis = Dis  # Distance matrix used to calcualte the EMD

        ssn_data.nBest = nBest  # Number of top best matches to keep
        ssn_data.bestTh = bestTh  # Variable that stores the nBest matches for each interval
        ssn_data.wAvI = wAvI  # Weighted threshold average based on the nBest matches for different intervals
        ssn_data.wSDI = wSDI  # Weighted threshold standard deviation based on the nBest matches for different intervals

        ssn_data.calRef = calRef  # Thresholded number of groups for reference that overlap with observer
        ssn_data.calObs = calObs  # Number of groups for observer that overlap with reference

        ssn_data.rSqI = rSqI  # R square of the y=x line for each separate interval
        ssn_data.mResI = mResI  # Mean residual of the y=x line for each separate interval
        ssn_data.rSqDT = rSqDT  # R square of the y=x line using the average threshold for each interval
        ssn_data.mResDT = mResDT  # Mean residual of the y=x line using the average threshold for each interval

        self.ssn_data = ssn_data
        # --------------------------------------------------------------------------------------------------------------

        print('done.', flush=True)
        print(' ', flush=True)

        return obsRefOvrlp

    def _mrange(self, min_values, max_values=None):
        """
            Inputs: min_values, a list/tuple with the starting values
                        if not given, assumed to be zero
                    max_values: a list/tuple with the ending values
            outputs: a tuple of values
        """

        if not max_values:
            max_values = min_values
            min_values = [0 for i in max_values]
        indices_list = copy(min_values)

        # Yield the (0,0, ..,0) value
        yield tuple(indices_list)

        while (True):
            indices_list = self._updateIndices(indices_list, min_values, max_values)
            if indices_list:
                yield tuple(indices_list)
            else:
                break  # We're back at the beginning

    def _updateIndices(self, indices_list, min_values, max_values):
        """
            Update the list of indices
        """
        for index in range(len(indices_list) - 1, -1, -1):

            # If the indices equals the max values, the reset it and
            # move onto the next value
            if not indices_list[index] == max_values[index] - 1:
                indices_list[index] += 1
                return indices_list
            else:
                indices_list[index] = min_values[index]
        return False

    def ADFsimultaneousEMD(self,
                           ssn_data,
                           disThres=1.25,
                           MaxIter=1000):

        """
        Function that peforms the EMD optimization by allowing variations of shift while keeping thresholds constant
        VARIABLES APPENDED TO THE OBJECT ARE SPECIFIED AT THE END

        :param disThres: Threshold above which we will ignore timeshifts (in units of the shortest
                         distance between observer and reference ADFs for each sub-interval separately)
        :param MaxIter:  Maximum number of iterations above which we skip simultaneous fit
        """

        print('Identify valid shifts for simultaneous fitting...', flush=True)

        # Dictionary that will store valid shift indices for each sub-interval
        valShfInx = []

        # Dictionary that will store the length of the index array for each sub-interval
        valShfLen = []

        # Going through different sub-intervals
        for siInx in range(0, ssn_data.cenPoints['OBS'].shape[0]):

            # Defining mask based on the interval type (rise or decay)
            if ssn_data.cenPoints['OBS'][siInx, 1] > 0:
                cadMaskI = ssn_data.risMask['INDEX']
            else:
                cadMaskI = ssn_data.decMask['INDEX']

            # Plot only if period is valid
            if ssn_data.vldIntr[siInx]:

                # Calculating minimum distance
                y = np.amin(ssn_data.EMDD[siInx], axis=0)

                # Appending valid indices to variable and storing length
                valShfInx.append((y <= disThres * np.min(y)).nonzero()[0])
                valShfLen.append(valShfInx[siInx].shape[0])

            # If period is not valid append ones so that they don't add to the permutations
            else:
                valShfInx.append(1)
                valShfLen.append(1)

        # Saving lengths as array
        valShfLen = np.array(valShfLen)

        print('Number of valid combinations:', np.nanprod(valShfLen))
        print(valShfLen)

        print('done.', flush=True)
        print(' ', flush=True)

        if np.nanprod(valShfLen) > MaxIter:
            ssn_data.disThres = np.nan  # Threshold above which we will ignore timeshifts
            ssn_data.EMDComb = np.nan  # Variable storing best simultaneous fits

            ssn_data.wAv = np.nan  # Weighted threshold average based on the nBest matches for all simultaneous fits
            ssn_data.wSD = np.nan  # Weighted threshold standard deviation based on the nBest matches for all simultaneous fits

            ssn_data.rSq = np.nan  # R square of the y=x line using a common threshold
            ssn_data.mRes = np.nan  # Mean residual of the y=x line using a common threshold

            ssn_data.rSqOO = np.nan  # R square of the y=x line using a common threshold, but only the valid intervals
            ssn_data.mResOO = np.nan  # Mean residual of the y=x line using a common threshold, but only the valid intervals

            return False

        print('Optimize EMD by varying shifts, but using the same threshold...', flush=True)

        # Allocating variable to store top matches
        EMDComb = np.ones((ssn_data.cenPoints['OBS'].shape[0] + 2, ssn_data.nBest)) * 10000

        # Identify first valid index
        fstVldIn = ssn_data.vldIntr.nonzero()[0][0]

        print('start', datetime.datetime.now(), '0 of', valShfLen[fstVldIn] - 1)

        comProg = 0
        for comb in self._mrange(valShfLen):

            # Inform user of progress
            if comb[fstVldIn] != comProg:
                print(comb[fstVldIn], 'of', valShfLen[fstVldIn] - 1, 'at', datetime.datetime.now())
                comProg = comb[fstVldIn]

                # Going through different thresholds for a given combination of shifts
            for TIdx in range(0, ssn_data.thN):

                # Initializing arrays for joining the ADFs of all sub-intervals
                ADFObsI = np.array([])
                ADFREFI = np.array([])

                # Joining ADF from all sub-interval for the specified shifts
                for siInx in range(0, ssn_data.cenPoints['OBS'].shape[0]):

                    # Append only if period is valid
                    if ssn_data.vldIntr[siInx]:

                        # If it is the first interval re-create the arrays
                        if ADFObsI.shape[0] == 0:

                            ADFObsI = np.divide(
                                ssn_data.GDObsI[siInx][TIdx, valShfInx[siInx][comb[siInx]], ssn_data.ODObsI[siInx][TIdx,
                                                                                            valShfInx[siInx][
                                                                                                comb[siInx]],
                                                                                            :] / ssn_data.MoLngt >= ssn_data.minObD],
                                ssn_data.ODObsI[siInx][TIdx, valShfInx[siInx][comb[siInx]], ssn_data.ODObsI[siInx][TIdx,
                                                                                            valShfInx[siInx][
                                                                                                comb[siInx]],
                                                                                            :] / ssn_data.MoLngt >= ssn_data.minObD])

                            ADFREFI = np.divide(
                                ssn_data.GDREFI[siInx][TIdx, valShfInx[siInx][comb[siInx]], ssn_data.ODREFI[siInx][TIdx,
                                                                                            valShfInx[siInx][
                                                                                                comb[siInx]],
                                                                                            :] / ssn_data.MoLngt >= ssn_data.minObD],
                                ssn_data.ODREFI[siInx][TIdx, valShfInx[siInx][comb[siInx]], ssn_data.ODREFI[siInx][TIdx,
                                                                                            valShfInx[siInx][
                                                                                                comb[siInx]],
                                                                                            :] / ssn_data.MoLngt >= ssn_data.minObD])

                        # If not, append ADF from all sub-interval for the specified shifts
                        else:
                            ADFObsI = np.append(ADFObsI, np.divide(
                                ssn_data.GDObsI[siInx][TIdx, valShfInx[siInx][comb[siInx]], ssn_data.ODObsI[siInx][TIdx,
                                                                                            valShfInx[siInx][
                                                                                                comb[siInx]],
                                                                                            :] / ssn_data.MoLngt >= ssn_data.minObD],
                                ssn_data.ODObsI[siInx][TIdx, valShfInx[siInx][comb[siInx]], ssn_data.ODObsI[siInx][TIdx,
                                                                                            valShfInx[siInx][
                                                                                                comb[siInx]],
                                                                                            :] / ssn_data.MoLngt >= ssn_data.minObD]))

                            ADFREFI = np.append(ADFREFI, np.divide(
                                ssn_data.GDREFI[siInx][TIdx, valShfInx[siInx][comb[siInx]], ssn_data.ODREFI[siInx][TIdx,
                                                                                            valShfInx[siInx][
                                                                                                comb[siInx]],
                                                                                            :] / ssn_data.MoLngt >= ssn_data.minObD],
                                ssn_data.ODREFI[siInx][TIdx, valShfInx[siInx][comb[siInx]], ssn_data.ODREFI[siInx][TIdx,
                                                                                            valShfInx[siInx][
                                                                                                comb[siInx]],
                                                                                            :] / ssn_data.MoLngt >= ssn_data.minObD]))

                # Calculating Earth Mover's Distance
                ADFObs, bins = np.histogram(ADFObsI, bins=(np.arange(0, ssn_data.MoLngt + 2) - 0.5) / ssn_data.MoLngt,
                                            density=True)
                ADFREF, bins = np.histogram(ADFREFI, bins=(np.arange(0, ssn_data.MoLngt + 2) - 0.5) / ssn_data.MoLngt,
                                            density=True)
                tmpEMD = emd(ADFREF.astype(np.float64), ADFObs.astype(np.float64), ssn_data.Dis.astype(np.float64))

                if np.any(EMDComb[0, :] > tmpEMD):

                    # Determining index for insertion
                    insInx = ssn_data.nBest - np.sum(EMDComb[0, :] >= tmpEMD)

                    # Initializing array to be inserted
                    insArr = [tmpEMD, TIdx]

                    # Append shifts
                    for siInx in range(0, ssn_data.cenPoints['OBS'].shape[0]):

                        # Append only if period is valid
                        if ssn_data.vldIntr[siInx]:
                            insArr.append(valShfInx[siInx][comb[siInx]])
                        # If not, append dummy
                        else:
                            insArr.append(np.nan)

                    # Convert to numpy array
                    insArr = np.array(insArr)

                    # Insert values
                    EMDComb = np.insert(EMDComb, insInx, insArr, axis=1)

                    # Remove last element
                    EMDComb = EMDComb[:, 0:ssn_data.nBest]

        print('done.', flush=True)
        print(' ', flush=True)

        print('Calculating average threshold and its standard deviation...', end="", flush=True)

        # Constructing weights
        alph = 1 - (EMDComb[0, :] - np.min(EMDComb[0, :])) / (np.max(EMDComb[0, :]) - np.min(EMDComb[0, :]))

        if np.isnan(np.sum(alph)):
            alph = EMDComb[0, :] * 0 + 1

        # Weighted average
        wAv = np.sum(np.multiply(alph, EMDComb[1, :])) / np.sum(alph)

        # Weighted Standard Deviation
        wSD = np.sqrt(np.sum(np.multiply(alph, np.power(EMDComb[1, :] - wAv, 2))) / np.sum(alph))

        print('done.', flush=True)
        print(' ', flush=True)

        rSq = np.nan
        mRes = np.nan

        print('Calculating r-square if there is overlap between observer and reference...', end="", flush=True)
        if (np.min(ssn_data.REF_Dat['ORDINAL']) <= np.min(ssn_data.ObsDat['ORDINAL'])) or (
                    np.max(ssn_data.REF_Dat['ORDINAL']) >= np.max(ssn_data.ObsDat['ORDINAL'])):

            # Calculating number of groups in reference data for given threshold
            grpsREFw = np.nansum(np.greater(ssn_data.REF_Dat.values[:, 3:ssn_data.REF_Dat.values.shape[1] - 3], wAv),
                                 axis=1).astype(float)
            grpsREFw[np.isnan(ssn_data.REF_Dat['AREA1'])] = np.nan

            # Selecting the days of overlap with calibrated observer
            grpsREFw = grpsREFw[np.in1d(ssn_data.REF_Dat['ORDINAL'].values, ssn_data.ObsDat['ORDINAL'].values)]
            grpsObsw = ssn_data.ObsDat.loc[
                np.in1d(ssn_data.ObsDat['ORDINAL'].values, ssn_data.REF_Dat['ORDINAL'].values), 'GROUPS'].values

            # Removing NaNs
            grpsREFw = grpsREFw[np.isfinite(grpsObsw)]
            grpsObsw = grpsObsw[np.isfinite(grpsObsw)]

            grpsObsw = grpsObsw[np.isfinite(grpsREFw)]
            grpsREFw = grpsREFw[np.isfinite(grpsREFw)]

            # Calculating goodness of fit of Y=X

            # R squared
            yMean = np.mean(grpsREFw)
            SStot = np.sum(np.power(grpsREFw - yMean, 2))
            SSreg = np.sum(np.power(grpsREFw - grpsObsw, 2))
            rSq = 1 - SSreg / SStot

            # Mean Residual
            mRes = np.mean(grpsREFw - grpsObsw)

            # Calculate R^2 and residual using only valid periods
            calRefN = np.array([0])
            calObsN = np.array([0])
            for n in range(0, ssn_data.cenPoints['OBS'].shape[0]):

                # Plot only if the period is valid and has overlap
                if ssn_data.vldIntr[n] and np.sum(
                        np.logical_and(ssn_data.REF_Dat['FRACYEAR'] >= ssn_data.endPoints['OBS'][n, 0],
                                       ssn_data.REF_Dat['FRACYEAR'] < ssn_data.endPoints['OBS'][
                                                   n + 1, 0])) > 0:
                    # Calculating number of groups in reference data for given threshold
                    grpsREFw = np.nansum(
                        np.greater(ssn_data.REF_Dat.values[:, 3:ssn_data.REF_Dat.values.shape[1] - 3], wAv),
                        axis=1).astype(float)
                    grpsREFw[np.isnan(ssn_data.REF_Dat['AREA1'])] = np.nan

                    # Selecting observer's interval
                    TObsDat = ssn_data.ObsDat.loc[
                        np.logical_and(ssn_data.ObsDat['FRACYEAR'] >= ssn_data.endPoints['OBS'][n, 0],
                                       ssn_data.ObsDat['FRACYEAR'] < ssn_data.endPoints['OBS'][n + 1, 0])
                        , 'GROUPS'].values.copy()
                    TObsOrd = ssn_data.ObsDat.loc[
                        np.logical_and(ssn_data.ObsDat['FRACYEAR'] >= ssn_data.endPoints['OBS'][n, 0],
                                       ssn_data.ObsDat['FRACYEAR'] < ssn_data.endPoints['OBS'][n + 1, 0])
                        , 'ORDINAL'].values.copy()

                    # Selecting the days of overlap with calibrated observer
                    grpsREFw = grpsREFw[np.in1d(ssn_data.REF_Dat['ORDINAL'].values, TObsOrd)]
                    grpsObsw = TObsDat[np.in1d(TObsOrd, ssn_data.REF_Dat['ORDINAL'].values)]

                    # Removing NaNs
                    grpsREFw = grpsREFw[np.isfinite(grpsObsw)]
                    grpsObsw = grpsObsw[np.isfinite(grpsObsw)]

                    grpsObsw = grpsObsw[np.isfinite(grpsREFw)]
                    grpsREFw = grpsREFw[np.isfinite(grpsREFw)]

                    # Appending to calibrated arrays?
                    calRefN = np.append(calRefN, grpsREFw)
                    calObsN = np.append(calObsN, grpsObsw)

            # Calculating goodness of fit of Y=X

            # R squared
            yMean = np.mean(calRefN)
            SStot = np.sum(np.power(calRefN - yMean, 2))
            SSreg = np.sum(np.power(calRefN - calObsN, 2))
            rSqOO = 1 - SSreg / SStot

            # Mean Residual
            mResOO = np.mean(calRefN - calObsN)

        print('done.', flush=True)
        print(' ', flush=True)

        # Storing variables in object-----------------------------------------------------------------------------------

        ssn_data.disThres = disThres  # Threshold above which we will ignore timeshifts
        ssn_data.EMDComb = EMDComb  # Variable storing best simultaneous fits

        ssn_data.wAv = wAv  # Weighted threshold average based on the nBest matches for all simultaneous fits
        ssn_data.wSD = wSD  # Weighted threshold standard deviation based on the nBest matches for all simultaneous fits

        ssn_data.rSq = rSq  # R square of the y=x line using a common threshold
        ssn_data.mRes = mRes  # Mean residual of the y=x line using a common threshold

        ssn_data.rSqOO = rSqOO  # R square of the y=x line using a common threshold, but only the valid intervals
        ssn_data.mResOO = mResOO  # Mean residual of the y=x line using a common threshold, but only the valid intervals

        # --------------------------------------------------------------------------------------------------------------

        print('done.', flush=True)
        print(' ', flush=True)

        return True
