from astropy import convolution as conv
from copy import copy
import datetime
import numpy as np
import os.path
import pandas as pd
from pyemd import emd
from scipy import signal
import scipy as sp
from scipy.ndimage.filters import gaussian_filter
import sys

from SSN_Input_Data import ssn_data
from SSN_Config import SSN_GRP_Config as config

parent_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'functions')
sys.path.insert(1, parent_dir)  # add to pythonpath
from detect_peaks import detect_peaks

class ssnGRP(ssn_data):
    """
    A class for managing SSN data, reference data, and performing GRP calculations
    """

    def __init__(self,
                 ref_data_path='input_data/SC_SP_RG_DB_KM_group_areas_by_day.csv',
                 silso_path='input_data/SN_m_tot_V2.0.csv',
                 silso_path_daily='input_data/SN_d_tot_V2.0.csv',
                 obs_data_path='../input_data/GNobservations_JV_V1.22.csv',
                 obs_observer_path='../input_data/GNobservers_JV_V1.22.csv',
                 output_path='output',
                 font=None,
                 minYrRef=1900,
                 dt=10,
                 phTol=2,
                 thS=5,
                 thE =130,
                 thI=1,
                 minObD=30,
                 maxValInt = 3,
                 GssKerPlt = 75,
                 plot=True):

        """
        Read all reference and observational and define the search parameters
        VARIABLES APPENDED TO THE OBJECT ARE SPECIFIED AT THE END

        :param ref_data_path: Location of the data to be used as reference
        :param silso_path: Location of silso's sunspot series
        :param obs_data_path: Location of the observational data
        :param obs_observer_path: Location of the file containing the observer's codes and names
        :param font: Font to be used while plotting
        :param minYrRef: Minimum year used for reference
        :param dt: Temporal Stride in days
        :param phTol: Cycle phase tolerance in years
        :param thS: Starting threshold
        :param thE: Ending threshold
        :param thI: Threshold increments
        :param minObD: Minimum number of days with non-zero groups to consider an interval valid
        :param maxValInt: Maximum number of valid intervals
        :param GssKerPlt: Number of days used on the gaussian kernel smoothing filter for plotting
        :param plot: Flag that enables the plotting and saving of relevant figures
        """

        if font is None:
            font = {'family': 'sans-serif', 'weight': 'normal', 'size': 21}

        # Use relative file paths even when running script from other directory
        dirname = os.path.dirname(__file__)
        ref_data_path = os.path.join(dirname, ref_data_path)
        silso_path = os.path.join(dirname, silso_path)
        silso_path_daily = os.path.join(dirname, silso_path_daily)
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
        REF_Dat = REF_Dat.loc[REF_Dat['YEAR'] >= minYrRef, :].reset_index(drop=True)


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
        Gss_1D_ker = conv.Gaussian1DKernel(GssKerPlt)
        REF_Grp['AVGROUPS'] = conv.convolve(REF_Grp['GROUPS'].values, Gss_1D_ker, preserve_nan=True)

        print('done.', flush=True)

        # ----------------------------------------------------------------------------------------------------
        print('Reading SILSO Data...', end="", flush=True)

        SILSO_Sn = pd.read_csv(silso_path, quotechar='"', encoding='utf-8', header=0)
        SILSO_Sn_d = pd.read_csv(silso_path_daily, quotechar='"', encoding='utf-8', header=0)

        # Including daily value and interpolating
        SILSO_Sn_d['MONTHSN'] = SILSO_Sn_d['DAILYSN']
        SILSO_Sn_d.loc[SILSO_Sn_d['DAILYSN'] < 0, 'MONTHSN'] = np.interp(
            SILSO_Sn_d.loc[SILSO_Sn_d['DAILYSN'] < 0, 'FRACYEAR'],
            SILSO_Sn_d.loc[SILSO_Sn_d['DAILYSN'] >= 0, 'FRACYEAR'],
            SILSO_Sn_d.loc[SILSO_Sn_d['DAILYSN'] >= 0, 'DAILYSN'])
        SILSO_Sn_d['DAILYSN'] = SILSO_Sn_d['DAILYSN'].astype(float)

        # Smoothing for plotting
        Gss_1D_ker = conv.Gaussian1DKernel(365)
        SILSO_Sn_d['AVGSNd'] = conv.convolve(SILSO_Sn_d['DAILYSN'].values, Gss_1D_ker, preserve_nan=True)

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

        # Building new REF_Grp with daily SN
        cond2 = pd.merge(SILSO_Sn_d, REF_Grp, on=['YEAR', 'MONTH', 'DAY'], how='inner')
        REF_Grp = REF_Grp.join(cond2['AVGSNd'])

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

        risMask = {'MASK': np.zeros(REF_Grp.shape[0], dtype=bool)}
        decMask = {'MASK': np.zeros(REF_Grp.shape[0], dtype=bool)}

        # Applying mask
        for cen in cenPointsR:
            if cen[1] == 1:
                risMask['MASK'][np.logical_and(REF_Grp['FRACYEAR'].values >= cen[0] - phTol,
                                               REF_Grp['FRACYEAR'].values <= cen[0] + phTol)] = True
            else:
                decMask['MASK'][np.logical_and(REF_Grp['FRACYEAR'].values >= cen[0] - phTol,
                                               REF_Grp['FRACYEAR'].values <= cen[0] + phTol)] = True

        # Creating cadence mask
        cadMask = np.zeros(REF_Grp.shape[0], dtype=bool)
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
        self.ssn_data.dt = dt  # Temporal Stride in days
        self.ssn_data.phTol = phTol  # Cycle phase tolerance in years
        self.ssn_data.thS = thS  # Starting threshold
        self.ssn_data.thE = thE  # Ending Threshold
        self.ssn_data.thI = thI  # Threshold increments
        self.ssn_data.Thresholds = np.arange(thS, thE+thI, thI) # Threshold to use in search

        self.ssn_data.GssKerPlt = GssKerPlt  # Number of days used on the gaussian kernel smoothing filter for plotting

        self.ssn_data.minObD = minObD  # Minimum number of days with non-zero groups to consider an interval valid
        self.ssn_data.maxValInt = maxValInt  # Maximum number of valid intervals

        self.ssn_data.REF_Dat = REF_Dat  # Reference data with individual group areas each day
        self.ssn_data.REF_Grp = REF_Grp  # Reference data with individual numbers of sunspot for each day
        self.ssn_data.SILSO_Sn_d = SILSO_Sn_d  # SILSO data for each day
        self.ssn_data.SILSO_Sn = SILSO_Sn  # Silso sunspot series
        self.ssn_data.SIL_max = SIL_max  # Maxima identified in the silso sunspot series
        self.ssn_data.SIL_min = SIL_min  # Minima identified in the silso sunspot series
        self.ssn_data.REF_min = REF_min  # Maxima identified in the reference data
        self.ssn_data.REF_max = REF_max  # Minima identified in the reference data

        self.ssn_data.risMask = risMask  # Mask indicating where to place the search window during raising phases
        self.ssn_data.decMask = decMask  # Mask indicating where to place the search window during declining phases

        self.ssn_data.endPoints = {
            'SILSO': endPointsS, 'REF': endPointsR}  # Variable that stores the boundaries of each rising and decaying phase
        self.ssn_data.cenPoints = {
            'SILSO': cenPointsS, 'REF': cenPointsR}  # Variable that stores the centers of each rising and decaying phase

        print('Done initializing data.', flush=True)
        print(' ', flush=True)

    # noinspection PyShadowingNames,PyShadowingNames
    def processObserver(self,
                        CalObs=412,
                        realThTol=0.05,
                        medianSw=True):
        """
        Function that breaks a given observer's data into "months", calculates the GRP and breaks it into rising and
        decaying intervals
        VARIABLES APPENDED TO THE OBJECT ARE SPECIFIED AT THE END

        :param CalObs: Observer identifier denoting observer to be processed
        :param realThTol: Propotionality tolerance for calculating real threshold
        :param medianSw: Switch that determines whether the error metrics are calculated on all points (False) or the medians of binned data (True))
        :return:  (False) True if there are (no) valid intervals
        """

        ssn_data = self.ssn_data

        NamObs = ssn_data.GN_Obs['OBSERVER'].values[ssn_data.GN_Obs['STATION'].values == CalObs]
        NamObs = NamObs[0]
        NamObs = NamObs[0:NamObs.find(',')].capitalize()

        print('Processing ' + NamObs, flush=True)

        # Picking observations
        ObsDat = ssn_data.GN_Dat[ssn_data.GN_Dat.STATION == CalObs].copy()
        ODObs = np.sum(np.isfinite(ObsDat), axis=1)

        # If no data for observer exit
        if ObsDat.shape[0] == 0:
            print('done. NO VALID INTERVALS IN OBSERVER', flush=True)
            print(' ', flush=True)
            ssn_data.CalObs = CalObs
            ssn_data.NamObs = NamObs
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

        # Removing repeated days
        u, indices = np.unique(ObsDat['ORDINAL'], return_index=True)
        ObsDat = ObsDat.iloc[indices, :].reset_index(drop=True)

        # Attaching daily SN to observer data
        cond1 = pd.merge(ssn_data.SILSO_Sn_d, ObsDat, on=['YEAR', 'MONTH', 'DAY'], how='inner')
        ObsDat = ObsDat.join(cond1['AVGSNd'])

        print('Calculating variables for plotting observer...', flush=True)

        # Average number of groups
        Gss_1D_ker = conv.Gaussian1DKernel(ssn_data.GssKerPlt)
        ObsDat['AVGROUPS'] = conv.convolve(ObsDat['GROUPS'].values, Gss_1D_ker, preserve_nan=True)


        optimThS = {'Low': np.nan, 'Optimal': np.nan, 'High': np.nan}

        # Finding overlap and appending it to Reference
        ssn_data.REF_Grp['OBSGROUPS'] = np.nan
        ssn_data.REF_Grp['AVOBSGROUPS'] = np.nan
        ssn_data.REF_Grp['GROUPSL'] =np.nan
        ssn_data.REF_Grp['GROUPSO'] =np.nan
        ssn_data.REF_Grp['GROUPSH'] =np.nan

        # Initializing centers, edges, and maximum in case there is no overlap
        maxNPlt = 5
        centers = np.nan
        edges = np.nan
        # If there is overlap between observer and reference
        if ((np.min(ssn_data.REF_Dat['ORDINAL']) <= np.min(ObsDat['ORDINAL'])) and (
                np.max(ssn_data.REF_Dat['ORDINAL']) >= np.min(ObsDat['ORDINAL']))) or (
                (np.max(ssn_data.REF_Dat['ORDINAL']) >= np.max(ObsDat['ORDINAL'])) and (
                np.min(ssn_data.REF_Dat['ORDINAL']) <= np.max(ObsDat['ORDINAL']))):

            cond1 = pd.merge(ssn_data.REF_Grp, ObsDat, on=['YEAR', 'MONTH', 'DAY'], how='left')
            ssn_data.REF_Grp['OBSGROUPS'] = cond1['GROUPS_y']

            # Imprinting common nan mask
            ssn_data.REF_Grp['INGROUPS'] = ssn_data.REF_Grp['GROUPS']
            ssn_data.REF_Grp.loc[np.isnan(ssn_data.REF_Grp['GROUPS']), 'OBSGROUPS'] = np.nan
            ssn_data.REF_Grp.loc[np.isnan(ssn_data.REF_Grp['OBSGROUPS']), 'INGROUPS'] = np.nan

            # Smoothing observer for comparison
            ssn_data.REF_Grp['AVOBSGROUPS'] = conv.convolve(ssn_data.REF_Grp['OBSGROUPS'].values, Gss_1D_ker, preserve_nan=True)

            # Calculating maximum for plotting, medians, and standard deviations
            maxNPlt = np.max([np.nanmax(ssn_data.REF_Grp['INGROUPS']), np.nanmax(ssn_data.REF_Grp['OBSGROUPS']), maxNPlt])

            # Number of bins to use
            Nbins = maxNPlt

            # Edges and Centers
            edges = np.arange(0, np.ceil(maxNPlt) + np.round(maxNPlt * 0.25), (np.ceil(maxNPlt)) / Nbins) - (
                np.ceil(maxNPlt)) / Nbins / 2
            centers = (edges[1:edges.shape[0]] + edges[0:edges.shape[0] - 1]) / 2

            # Applying Sqrt + 1
            if config.SQRT_2DHIS:
                maxNPlt = np.sqrt(maxNPlt)

                # Edges and Centers
                edges = np.arange(1, np.ceil(maxNPlt) * 1.05, (np.ceil(maxNPlt)) / Nbins) - (np.ceil(maxNPlt)) / Nbins / 2
                centers = (edges[1:edges.shape[0]] + edges[0:edges.shape[0] - 1]) / 2

            # Going through different thresholds to find "real" threshold
            optimTh = np.zeros((ssn_data.Thresholds.shape[0], 5))*np.nan
            optimTh[:, 0] = ssn_data.Thresholds
            for TIdx, Thr in enumerate(ssn_data.Thresholds):

                # Thresholded Ref Groups
                Grp_CompR = np.nansum(
                    np.greater(ssn_data.REF_Dat.values[
                               :, 3:ssn_data.REF_Dat.values.shape[1] - 2], Thr), axis=1).astype(float)
                Grp_CompO = ssn_data.REF_Grp['OBSGROUPS'].copy()


                # Imprinting Valid Interval NaNs
                nanmsk = np.isnan(Grp_CompO)
                Grp_CompR[nanmsk] = np.nan
                nanmsk = np.isnan(Grp_CompR)
                Grp_CompO[nanmsk] = np.nan

                # Plotting variable
                metricsDic = self.Calculate_R2M_MRes_MRRes(Grp_CompO, Grp_CompR, centers, edges, medianSw=medianSw)

                # Calculating average ratio between groups
                optimTh[TIdx, 1] = metricsDic['Slope']
                # # Calculate average difference between groups
                # optimTh[TIdx, 2] = (np.nanmean(Grp_CompR - ssn_data.REF_Grp['OBSGROUPS']))
                #
                # # Smoothing
                # Grp_CompR = conv.convolve(Grp_CompR, Gss_1D_ker, preserve_nan=True)
                # # Calculating average ratio between smoothed series
                # optimTh[TIdx, 3] = np.nanmean(Grp_CompR/ssn_data.REF_Grp['AVOBSGROUPS'])
                #
                # # Calculate average difference between smoothed series
                # optimTh[TIdx, 4] = (np.nanmean(Grp_CompR - ssn_data.REF_Grp['AVOBSGROUPS']))


            # Storing range of optimal match and Smoothing for plotting
            optimThS['Low'] = optimTh[np.abs(optimTh[:, 1]-(1-realThTol)) == np.min(np.abs(optimTh[:, 1]-(1-realThTol))), 0][0]
            ssn_data.REF_Grp['GROUPSL'] = np.nansum(np.greater(ssn_data.REF_Dat.values[:, 3:ssn_data.REF_Dat.values.shape[1] - 2], optimThS['Low']), axis=1)
            ssn_data.REF_Grp['GROUPSL'] = ssn_data.REF_Grp['GROUPSL'].astype(float)
            ssn_data.REF_Grp.loc[np.isnan(ssn_data.REF_Dat['AREA1']), 'GROUPSL'] = np.nan
            ssn_data.REF_Grp['GROUPSL'] = conv.convolve(ssn_data.REF_Grp['GROUPSL'].values, Gss_1D_ker)

            optimThS['Optimal'] = optimTh[np.abs(optimTh[:, 1]-1) == np.min(np.abs(optimTh[:, 1]-1)), 0][0]
            ssn_data.REF_Grp['GROUPSO'] = np.nansum(np.greater(ssn_data.REF_Dat.values[:, 3:ssn_data.REF_Dat.values.shape[1] - 2], optimThS['Optimal']), axis=1)
            ssn_data.REF_Grp['GROUPSO'] = ssn_data.REF_Grp['GROUPSO'].astype(float)
            ssn_data.REF_Grp.loc[np.isnan(ssn_data.REF_Dat['AREA1']), 'GROUPSO'] = np.nan
            ssn_data.REF_Grp['GROUPSO'] = conv.convolve(ssn_data.REF_Grp['GROUPSO'].values, Gss_1D_ker)

            optimThS['High'] = optimTh[np.abs(optimTh[:, 1]-(1+realThTol)) == np.min(np.abs(optimTh[:, 1]-(1+realThTol))), 0][0]
            ssn_data.REF_Grp['GROUPSH'] = np.nansum(np.greater(ssn_data.REF_Dat.values[:, 3:ssn_data.REF_Dat.values.shape[1] - 2], optimThS['High']), axis=1)
            ssn_data.REF_Grp['GROUPSH'] = ssn_data.REF_Grp['GROUPSH'].astype(float)
            ssn_data.REF_Grp.loc[np.isnan(ssn_data.REF_Dat['AREA1']), 'GROUPSH'] = np.nan
            ssn_data.REF_Grp['GROUPSH'] = conv.convolve(ssn_data.REF_Grp['GROUPSH'].values, Gss_1D_ker)

        # Finding internal endpoints and centers of Observer Intervals are included if their center is covered by the observer
        # Defining boolean array of valid endpoints
        validEnd = np.logical_and(ssn_data.endPoints['SILSO'][:, 0] > np.min(ObsDat['FRACYEAR']),
                                  ssn_data.endPoints['SILSO'][:, 0] < np.max(ObsDat['FRACYEAR']))

        # Adding a True on the index prior to the first endpoint to include the bracketing point
        validEnd[0:validEnd.shape[0] - 2] = np.logical_or(validEnd[0:validEnd.shape[0] - 2],
                                                          validEnd[1:validEnd.shape[0] - 1])
        # Adding a True on the index after the last endpoint to include the bracketing point
        validEnd[2:validEnd.shape[0] - 0] = np.logical_or(validEnd[2:validEnd.shape[0]],
                                                          validEnd[1:validEnd.shape[0] - 1])

        # Defining arrays
        endPoints = ssn_data.endPoints['SILSO'][validEnd, :]

        if endPoints.shape[0] == 0:
            endPoints = np.array([[np.min(ObsDat['FRACYEAR']), 1], [np.max(ObsDat['FRACYEAR']), -1]])

        cenPoints = (endPoints[0:len(endPoints) - 1, :] + endPoints[1:len(endPoints), :]) / 2
        cenPoints[:, 1] = endPoints[1:endPoints.shape[0], 1]

        # Identification of Min-Max Max-Min intervals with enough valid days
        vldIntr = np.zeros(cenPoints.shape[0])

        # Number of days rising or declining
        rise_count = 0
        dec_count = 0

        # Creating Storing list
        optimThM = []

        for siInx in range(0, cenPoints.shape[0]):

            # Redefining endpoints if interval is partial
            if endPoints[siInx, 0] < np.min(ObsDat['FRACYEAR']):
                print('Redefining left endpoint')
                endPoints[siInx, 0] = np.min(ObsDat['FRACYEAR'])

            if endPoints[siInx + 1, 0] > np.max(ObsDat['FRACYEAR']):
                print('Redefining right endpoint')
                endPoints[siInx + 1, 0] = np.max(ObsDat['FRACYEAR'])

            print('Center:', np.round(cenPoints[siInx, 0], 2), 'Edges:', np.round(endPoints[siInx, 0], 2),
                  np.round(endPoints[siInx + 1, 0], 2))

            # Selecting interval
            TObsDat = ObsDat.loc[np.logical_and(ObsDat['FRACYEAR'] >= endPoints[siInx, 0],
                                                ObsDat['FRACYEAR'] < endPoints[siInx + 1, 0]), 'GROUPS'].values.copy()
            NgoodDays = np.sum(TObsDat > 0)
            if NgoodDays > ssn_data.minObD:
                # Marking interval as valid
                vldIntr[siInx] = np.sum(TObsDat > 0)
                print('Valid interval. The number of with more than one group is : ', NgoodDays)

                # Calculating observer metrics and properties
                if cenPoints[siInx, 1] == 1.0:
                    rise_count += np.sum(np.isfinite(TObsDat))
                elif cenPoints[siInx, 1] == -1.0:
                    dec_count += np.sum(np.isfinite(TObsDat))

                # Finding optimal threshold
                optimThT = {'Low': np.nan, 'Optimal': np.nan, 'High': np.nan}

                # Check if Observer has overlap with reference
                ovrlpmsk = np.logical_and(ssn_data.REF_Dat['FRACYEAR'] > endPoints[siInx, 0],
                                         ssn_data.REF_Dat['FRACYEAR'] < endPoints[siInx + 1, 0])
                if np.sum(ovrlpmsk) > 0:

                    # Going through different thresholds to find "real" threshold
                    optimTh = np.zeros((ssn_data.Thresholds.shape[0], 2)) * np.nan
                    optimTh[:, 0] = ssn_data.Thresholds
                    for TIdx, Thr in enumerate(ssn_data.Thresholds):
                        # Thresholded Ref Groups
                        Grp_CompR = np.nansum(
                            np.greater(ssn_data.REF_Dat.values[
                                       :, 3:ssn_data.REF_Dat.values.shape[1] - 2], Thr), axis=1).astype(float)
                        Grp_CompO = ssn_data.REF_Grp['OBSGROUPS'].copy()

                        # Imprinting Valid Interval NaNs
                        nanmsk = np.isnan(ssn_data.REF_Grp['OBSGROUPS'])
                        nanmsk = np.logical_or(nanmsk, np.logical_not(ovrlpmsk))
                        Grp_CompR[nanmsk] = np.nan
                        Grp_CompO[nanmsk] = np.nan

                        metricsDic = self.Calculate_R2M_MRes_MRRes(Grp_CompO, Grp_CompR, centers, edges, medianSw=medianSw)

                        # Calculating average ratio between groups
                        optimTh[TIdx, 1] = metricsDic['Slope']

                    # Storing range of optimal match and Smoothing for plotting
                    optimThT['Low'] = optimTh[
                        np.abs(optimTh[:, 1] - (1 - realThTol)) == np.min(np.abs(optimTh[:, 1] - (1 - realThTol))), 0][0]
                    optimThT['Optimal'] = optimTh[np.abs(optimTh[:, 1] - 1) == np.min(np.abs(optimTh[:, 1] - 1)), 0][0]
                    optimThT['High'] = optimTh[
                        np.abs(optimTh[:, 1] - (1 + realThTol)) == np.min(np.abs(optimTh[:, 1] - (1 + realThTol))), 0][0]

                optimThM.append(optimThT)

            else:
                optimThM.append([])
                print('INVALID interval. The number of with more than one group is: ', NgoodDays)

            print(' ')

        print(str(np.sum(vldIntr>0)) + '/' + str(vldIntr.shape[0]) + ' valid intervals')

        # If more than the maximum, retain only the intervals with the largest number of days
        if np.sum(vldIntr > 0) > ssn_data.maxValInt:
            I = np.argsort(vldIntr)[::-1]
            vldIntr[vldIntr < vldIntr[I][ssn_data.maxValInt-1]] = np.nan

        # Remove invalid intervals
        vldIntr[vldIntr == 0] = np.nan

        # Storing variables in object-----------------------------------------------------------------------------------

        ssn_data.CalObs = CalObs  # Observer identifier denoting observer to be processed
        ssn_data.NamObs = NamObs  # Name of observer
        ssn_data.ObsDat = ObsDat  # Data of observer being analyzed

        ssn_data.medianSw = medianSw  # Switch that determines whether the error metrics are calculated on all points (False) or the medians of binned data (True))

        ssn_data.endPoints['OBS'] = endPoints  # Variable that stores the boundaries of each rising and decaying phase
        ssn_data.cenPoints['OBS'] = cenPoints  # Variable that stores the centers of each rising and decaying phase

        ssn_data.vldIntr = vldIntr  # Stores whether each rising or decaying interval has enough data to be valid

        ssn_data.optimThM = optimThM  # Multi-threshold optimal match
        ssn_data.optimThS = optimThS  # Single-threshold optimal match

        ssn_data.maxNPlt = maxNPlt  # Maximum value of groups for plotting and calculation of standard deviations
        ssn_data.centers = centers  # Centers of the bins used to plot and calculate r square
        ssn_data.edges = edges  # Centers of the bins used to plot and calculate r square

        # NEW MACHINE LEARNING PARAMETERS
        ssn_data.ODObs = ODObs  # NUMBER OF DAYS WITH OBSERVATIONS

        n_qui = ObsDat.loc[ObsDat.GROUPS == 0.0].shape[0]
        n_act = np.isfinite(ObsDat.loc[ObsDat.GROUPS > 0.0]).shape[0]
        n_na = ObsDat.loc[np.isnan(ObsDat.GROUPS)].shape[0]
        ssn_data.QDays = n_qui  # Total number of Quiet days
        ssn_data.ADays = n_act  # Total number of Active days
        ssn_data.NADays = n_na  # Total number of missing days in data
        ssn_data.QAFrac = round(n_qui / n_act, 3)  # Fraction of quiet to active days

        ssn_data.RiseDays = rise_count  # Number of days in rising phases
        ssn_data.DecDays = dec_count  # Number of days in declining phases

        ssn_data.ObsStartDate = ObsDat['ORDINAL'].data[0]
        ssn_data.ObsTotLength = ObsDat['ORDINAL'].data[-1] - ObsDat['ORDINAL'].data[0]

        # --------------------------------------------------------------------------------------------------------------

        # Create folder for observer's output
        if not os.path.exists(ssn_data.output_path + '/' + str(CalObs) + '_' + NamObs):
            os.makedirs(ssn_data.output_path + '/' + str(CalObs) + '_' + NamObs)

        if np.nansum(vldIntr) == 0:
            print('done. NO VALID INTERVALS IN OBSERVER', flush=True)
            print(' ', flush=True)
            return False
        else:
            print('done. Observer has valid intervals', flush=True)
            print(' ', flush=True)

            return True

    @staticmethod
    def Calculate_R2M_MRes_MRRes(calObsT,
                                  calRefT,
                                  centers,
                                  edges,
                                  medianSw=True):

        """
        Function that calculates the R^2 and mean residuals using the medians of binned data.

        :param calObsT: Number of groups per day for the calibrated observer
        :param centers: Centers of the bins used in the calibration
        :param edges: Edges of the bins used in the calibration
        :param medianSw: Switch that determines whether the error metrics are calculated on all points (False) or the medians of binned data (True))
        """

        # Applying Sqrt + 1
        if config.SQRT_2DHIS:
            calRefT = np.sqrt(calRefT + 1)
            calObsT = np.sqrt(calObsT + 1)

        # Calculating Median for bins
        Ymedian = centers * np.nan

        for i in range(0, centers.shape[0]):
            ypoints = calRefT[np.logical_and(calObsT >= edges[i], calObsT <= edges[i + 1])]
            if ypoints.shape[0] > 0:
                Ymedian[i] = np.nanmedian(ypoints)

        # Calculating quantities for assessment using medians
        if medianSw:


            y = Ymedian
            x = centers

            x = x[np.isfinite(y)]
            y = y[np.isfinite(y)]

            # R squared
            yMean = np.mean(y)
            SStot = np.sum(np.power(y - yMean, 2))
            SSreg = np.sum(np.power(y - x, 2))
            rSq = (1 - SSreg / SStot)

            # Mean Residual
            mRes = np.mean(y - x)
            mRRes = np.mean(np.divide(y[x > 0] - x[x > 0], x[x > 0]))

            # Slope
            y = y[x > 0]
            x = x[x > 0]
            Slope = np.nanmean(y / x)

        # Calculating quantities for assessment using all points
        else:

            y = calRefT
            x = calObsT

            x = x[np.isfinite(y)]
            y = y[np.isfinite(y)]

            # R squared
            yMean = np.mean(y)
            SStot = np.sum(np.power(y - yMean, 2))
            SSreg = np.sum(np.power(y - x, 2))
            rSq = (1 - SSreg / SStot)

            # Mean Residual
            mRes = np.mean(y - x)
            # Mean Relative Residual
            mRRes = np.mean(np.divide(y[x > 0] - x[x > 0], x[x > 0]))

            y = y[x > 0]
            x = x[x > 0]
            Slope = np.nanmean(y/x)

        return {'rSq': rSq,
                'mRes': mRes,
                'mRRes': mRRes,
                'Slope': Slope}

    def GRPscanningWindowDis(self,
                             noOvrlpSw=True,
                             onlyActiveSw=True,
                             emdSw=True,
                             Dis_Pow=2,
                             NGrpsBins=26,
                             nBest1=100,
                             nBest2=10,
                             sigmaTh=0.5,
                             sigmaT=2.5):

        """
        Function that preps the search windows and calculates the EMD for each separate rising and decaying interval
        comparing the observer and the reference
        VARIABLES APPENDED TO THE OBJECT ARE SPECIFIED AT THE END

        :input
        :param noOvrlpSw: Switch that forces the code to ignore the true overlapping phase in calibration if present
        :param onlyActiveSw: Switch that ignores (True) or includes zeros (False)
        :param emdSw: Switch that activates the EMD metric (True), vs the L2 norm (False)
        :param Dis_Pow: Power index used to define the distance matrix for EMD calculation
        :param NGrpsBins: Number of used bins to calculate the group distribution
        :param nBest1: Number of best matches to keep based on optimal distance
        :param nBest2: Number of best matches to keep based on similarity in number of observations
        :param sigmaTh: Sigma in threshold of the smoothing gaussian filter applied to the optimization matrix
        :param sigmaT: Sigma in days of the smoothing gaussian filter applied to the optimization matrix
        :return:  (False) True if there are (no) valid days of overlap between observer and reference
        """

        print('Calculating the Earths Mover Distance using a sliding window...', flush=True)

        ssn_data = self.ssn_data

        # Reducing number of valid intervals to maximum allowed.  This is done here so that the ignored intervals are marked clearly in the plots
        ssn_data.vldIntr[np.isnan(ssn_data.vldIntr)] = 0

        # Setting the bin edges for EMD calculation
        EMDbinsG = (np.arange(onlyActiveSw, NGrpsBins + 2) - 0.5)

        # Distance matrix for EMD calculations
        x = np.arange(0, NGrpsBins + 2 - onlyActiveSw)
        y = np.arange(0, NGrpsBins + 2 - onlyActiveSw)
        xx, yy = np.meshgrid(x, y)
        DisG = np.absolute(np.power(xx - yy, Dis_Pow))

        # Creating Storing dictionaries
        # Distance between distributions of sunspot groups for interval
        EMDtD = []
        EMDtDi = []
        EMDthD = []
        EMDthDi = []
        EMDGr = []

        # Distributions
        grpDirObsI = []
        grpDirREFI = []

        # Number of observations
        grpNObsI = []
        grpNREFI = []

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
                                   ssn_data.ObsDat['FRACYEAR'] < ssn_data.endPoints['OBS'][siInx + 1, 0]),
                    'GROUPS'].values.copy()

                TObsFYr = ssn_data.ObsDat.loc[
                    np.logical_and(ssn_data.ObsDat['FRACYEAR'] >= ssn_data.endPoints['OBS'][siInx, 0],
                                   ssn_data.ObsDat['FRACYEAR'] < ssn_data.endPoints['OBS'][siInx + 1, 0])
                    , 'FRACYEAR'].values.copy()

                # Find index of center of sub-interval
                minYear = np.min(np.absolute(TObsFYr - ssn_data.cenPoints['OBS'][siInx, 0]))
                obsMinInx = (np.absolute(TObsFYr - ssn_data.cenPoints['OBS'][siInx, 0]) == minYear).nonzero()[0][0]

                # Creating Storing Variables

                # EMD, times, and thresholds
                EMDGrt = np.ones((ssn_data.Thresholds.shape[0], cadMaskI.shape[0])) * 1e16
                EMDt = np.zeros((ssn_data.Thresholds.shape[0], cadMaskI.shape[0]))
                EMDti = np.zeros((ssn_data.Thresholds.shape[0], cadMaskI.shape[0]))
                EMDth = np.zeros((ssn_data.Thresholds.shape[0], cadMaskI.shape[0]))
                EMDthi = np.zeros((ssn_data.Thresholds.shape[0], cadMaskI.shape[0]))

                # Number of non-zero observations
                grpNObs = np.zeros((ssn_data.Thresholds.shape[0], cadMaskI.shape[0]))
                grpNREF = np.zeros((ssn_data.Thresholds.shape[0], cadMaskI.shape[0]))

                # Group distributions
                grpDirObs = np.zeros((ssn_data.Thresholds.shape[0], cadMaskI.shape[0], EMDbinsG.shape[0]-1))
                grpDirREF = np.zeros((ssn_data.Thresholds.shape[0], cadMaskI.shape[0], EMDbinsG.shape[0]-1))

                # Going through different thresholds
                for TIdx, Thr in enumerate(ssn_data.Thresholds[0:-1]):

                    # Calculating number of groups in reference data for given threshold
                    grpsREFw = np.nansum(
                        np.greater(ssn_data.REF_Dat.values[:, 3:ssn_data.REF_Dat.values.shape[1] - 2],
                                   Thr),
                        axis=1).astype(float)
                    grpsREFw[np.isnan(ssn_data.REF_Dat['AREA1'])] = np.nan

                    # Going through different shifts
                    for SIdx in range(0, cadMaskI.shape[0]):

                        # Calculating bracketing indices
                        Idx1 = cadMaskI[SIdx] - obsMinInx
                        Idx2 = Idx1 + TObsDat.shape[0]

                        # Making sure there is no observational overlap between observer and reference
                        if np.logical_or(np.logical_not(noOvrlpSw),
                                         np.logical_or(np.min(ssn_data.REF_Dat['FRACYEAR'][Idx1:Idx2]) > np.max(TObsFYr),
                                                       np.max(ssn_data.REF_Dat['FRACYEAR'][Idx1:Idx2]) < np.min(TObsFYr))):

                            # Selecting reference window of matching size to observer sub-interval;
                            TgrpsREF = grpsREFw[Idx1:Idx2].copy()
                            TgrpsOb = TObsDat.copy()

                            # Making sure selections have the same length
                            if TgrpsREF.shape[0] == TgrpsOb.shape[0]:

                                # Imprinting missing days
                                # OBSERVER
                                TgrpsOb[np.isnan(TgrpsREF)] = np.nan
                                # REFERENCE
                                TgrpsREF[np.isnan(TgrpsOb)] = np.nan

                                # Retaining only non-nans and/or non-zeros
                                if onlyActiveSw:
                                    TgrpsOb = TgrpsOb[TgrpsOb > 0]
                                    TgrpsREF = TgrpsREF[TgrpsREF > 0]
                                else:
                                    TgrpsOb = TgrpsOb[np.isfinite(TgrpsOb)]
                                    TgrpsREF = TgrpsREF[np.isfinite(TgrpsREF)]

                                # Number of observations
                                grpNObs[TIdx, SIdx] = np.sum(TgrpsOb>0)
                                grpNREF[TIdx, SIdx] = np.sum(TgrpsREF>0)

                                # Calculating Earth Mover's Distance for group numbers
                                DisObs, bins = np.histogram(TgrpsOb, bins=EMDbinsG,
                                                            density=True)
                                DisREF, bins = np.histogram(TgrpsREF, bins=EMDbinsG,
                                                            density=True)

                                # Storing Distance
                                if emdSw:
                                    EMDGrt[TIdx, SIdx] = emd(DisREF.astype(np.float64), DisObs.astype(np.float64),
                                                             DisG.astype(np.float64))
                                else:
                                    EMDGrt[TIdx,SIdx] = np.nanmean(np.power(DisObs-DisREF,2))

                                # Storing Distributions
                                grpDirObs[TIdx, SIdx, :] = DisObs
                                grpDirREF[TIdx, SIdx, :] = DisREF

                                # Storing coordinates of EMD distances
                                EMDt[TIdx, SIdx] = ssn_data.REF_Grp['FRACYEAR'].values[cadMaskI[SIdx]]
                                EMDti[TIdx, SIdx] = SIdx
                                EMDth[TIdx, SIdx] = Thr
                                EMDthi[TIdx, SIdx] = TIdx


            # If period is not valid append empty variavbles
            else:
                print('[{}/{}] INVALID Interval'.format(siInx + 1, num_intervals))
                EMDGrt = []
                EMDt = []
                EMDti = []
                EMDth = []
                EMDthi = []
                grpDirObs = []
                grpDirREF = []
                grpNObs = []
                grpNREF = []

            print(' ')

            # Appending variables
            EMDGr.append(EMDGrt)
            EMDtD.append(EMDt)
            EMDtDi.append(EMDti)
            EMDthD.append(EMDth)
            EMDthDi.append(EMDthi)
            grpDirObsI.append(grpDirObs)
            grpDirREFI.append(grpDirREF)
            grpNObsI.append(grpNObs)
            grpNREFI.append(grpNREF)

        print('done.', flush=True)
        print(' ', flush=True)

        print('Identifying the best matches for each valid period and looking for ref-obs overlap...', end="",
              flush=True)

        # Creating Storing dictionaries to store best thresholds
        bestTh  = []  # nBest1 best thresholds based only on distribution distance
        bestTh2 = []  # nBest2 best thresholds also using relative difference in number of active days
        calRef = []
        calObs = []

        # Color axis limits
        minGrp = 1e10
        maxGrp = 0
        maxGrp2 = 0

        # Switch indicating that there is overlap between reference and observer
        obsRefOvrlp = False

        # Variables to store the best threshold for each interval
        bThI = ssn_data.vldIntr.copy() * 0.0

        for siInx in range(0, ssn_data.cenPoints['OBS'].shape[0]):

            # Analyze period only if valid
            if ssn_data.vldIntr[siInx]:

                # Smooth EMD (threshold, cadence)
                EMDGr[siInx] = gaussian_filter(EMDGr[siInx], (sigmaTh/ssn_data.thI, sigmaT/ssn_data.dt), mode='reflect')

                # Creating matrix for sorting and find the best combinations of threshold and shift
                OpMat = np.concatenate((EMDtD[siInx].reshape((-1, 1)),  # Shift
                                        EMDthD[siInx].reshape((-1, 1)),  # Threshold
                                        EMDGr[siInx].reshape((-1, 1)),  # Distance
                                        np.abs(grpNObsI[siInx].reshape((-1, 1)) - grpNREFI[siInx].reshape((-1, 1))) /
                                        grpNObsI[siInx].reshape((-1, 1)),  # Relative difference in active days
                                        EMDtDi[siInx].reshape((-1, 1)),  # Time shift index
                                        EMDthDi[siInx].reshape((-1, 1))),  # Threshold index
                                        axis=1)

                # Sort according to EMD to find the best matches
                I = np.argsort(OpMat[:, 2], axis=0)
                OpMat = np.squeeze(OpMat[I, :])

                # Caculate extrema for plotting
                minGrp = np.nanmin([minGrp, OpMat[0, 2]])
                maxGrp = np.nanmax([maxGrp, np.nanpercentile(OpMat[:, 2], 10)])

                # Grab the best matches
                OpMat = OpMat[0:nBest1, :]

                # Sort according to EMD
                I = np.argsort(OpMat[:, 2], axis=0)
                OpMat = np.squeeze(OpMat[I, :])

                # Adding best points
                bestTh.append(OpMat)

                # Caculate extrema for plotting
                maxGrp2 = np.nanmax([maxGrp2, np.nanmax(OpMat[:, 2])])

                # Sort according to number of non-zero observations
                I = np.argsort(OpMat[:, 3], axis=0)
                OpMat = np.squeeze(OpMat[I, :])

                # Grab the best matches
                OpMat = OpMat[0:nBest2, :]

                # Sort according to EMD
                I = np.argsort(OpMat[:, 2], axis=0)
                OpMat = np.squeeze(OpMat[I, :])

                bestTh2.append(OpMat)

                # Best Threshold
                bThI[siInx] = OpMat[0, 1]

                # Check if Observer has overlap with reference
                if np.sum(np.logical_and(ssn_data.REF_Dat['FRACYEAR'] > ssn_data.endPoints['OBS'][siInx, 0],
                                         ssn_data.REF_Dat['FRACYEAR'] < ssn_data.endPoints['OBS'][siInx + 1, 0])) > 0:

                    # Activate the overlap switch
                    obsRefOvrlp = True

                    # Calculating number of groups in reference data for given threshold
                    grpsREFw = np.nansum(
                        np.greater(ssn_data.REF_Dat.values[:, 3:ssn_data.REF_Dat.values.shape[1] - 2], bThI[siInx]),
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

                    calRef.append(grpsREFw)
                    calObs.append(grpsObsw)

                else:
                    calRef.append([])
                    calObs.append([])

            # If period not valid store an empty array
            else:
                bestTh.append([])
                bestTh2.append([])
                calRef.append([])
                calObs.append([])

        # Creating storing lists to store fit properties
        rSqI = []
        mResI = []
        mRResI = []
        slopeI = []

        for siInx in range(0, ssn_data.cenPoints['OBS'].shape[0]):

            # Analyze period only if valid
            if ssn_data.vldIntr[siInx]:

                # Calculate goodness of fit if overlap is true
                if obsRefOvrlp and len(calRef[siInx]) > 0:

                    calRefT = calRef[siInx].copy()
                    calObsT = calObs[siInx].copy()

                    metricsDic = self.Calculate_R2M_MRes_MRRes(calObsT, calRefT, ssn_data.centers, ssn_data.edges, medianSw=ssn_data.medianSw)

                    rSqI.append(metricsDic['rSq'])
                    mResI.append(metricsDic['mRes'])
                    mRResI.append(metricsDic['mRRes'])
                    slopeI.append(metricsDic['Slope'])

                else:

                    rSqI.append([])
                    mResI.append([])
                    mRResI.append([])
                    slopeI.append([])

            # If period not valid store an empty array
            else:
                rSqI.append([])
                mResI.append([])
                mRResI.append([])
                slopeI.append([])

        # Metrics dictionary for different threshold calculations
        mDDT = {'rSq': np.nan,
                'mRes': np.nan,
                'mRRes': np.nan,
                'Slope': np.nan}

        # Only if there is at least only one interval that is valid
        if len(calRef) > 0 and obsRefOvrlp:
            calRefT = np.concatenate(calRef, axis=0)
            calObsT = np.concatenate(calObs, axis=0)

            mDDT = self.Calculate_R2M_MRes_MRRes(calObsT, calRefT, ssn_data.centers, ssn_data.edges, medianSw=ssn_data.medianSw)

        # Storing variables in object-----------------------------------------------------------------------------------
        ssn_data.noOvrlpSw = noOvrlpSw   # Switch that forces the code to ignore the true overlapping phase in calibration if present
        ssn_data.onlyActiveSw = onlyActiveSw  # Switch that ignores (True) or includes zeros (False)
        ssn_data.emdSw = emdSw  # Switch that activates the EMD metric (True), vs the L2 norm (False)

        ssn_data.NGrpsBins = NGrpsBins  # Number of used bins to calculate the group distribution

        ssn_data.nBest1 = nBest1  # Number of best matches to keep based on optimal distance
        ssn_data.nBest2 = nBest2  # Number of best matches to keep based on similarity in number of observations

        ssn_data.EMDbins = EMDbinsG # Bin edges for group EMD calculation
        ssn_data.EMDcaxis = np.array([minGrp, maxGrp, maxGrp2])  # Color axis limits for EMD plot
        ssn_data.Dis = DisG  # Distance matrix used to calcualte the group EMD

        ssn_data.EMDD = EMDGr  # Variable that stores the EMD between the reference and the observer for each interval, threshold, and window shift
        ssn_data.EMDtD = EMDtD  # Variable that stores the windowshift matching EMDD for each interval, threshold, and window shift
        ssn_data.EMDthD = EMDthD  # Variable that stores the threshold matching EMDD for each interval, threshold, and window shift

        # Distributions
        ssn_data.grpDirObsI = grpDirObsI
        ssn_data.grpDirREFI = grpDirREFI

        # Number of observations
        ssn_data.grpNObsI = grpNObsI
        ssn_data.grpNREFI = grpNREFI

        ssn_data.bestTh  = bestTh  # nBest1 best thresholds based only on distribution distance for each interval
        ssn_data.bestTh2 = bestTh2  # nBest2 best thresholds also using relative difference in number of active days for each interval

        ssn_data.bThI = bThI  # Weighted threshold average based on the nBest matches for different intervals

        ssn_data.calRef = calRef  # Thresholded number of groups for reference that overlap with observer
        ssn_data.calObs = calObs  # Number of groups for observer that overlap with reference

        ssn_data.rSqI = rSqI  # R square of the y=x line for each separate interval
        ssn_data.mResI = mResI  # Mean residual of the y=x line for each separate interval
        ssn_data.mRResI = mRResI  # Mean relative residual of the y=x line for each separate interval
        ssn_data.slopeI = slopeI  # Slope of y/x

        ssn_data.mDDT = mDDT  # Metrics dictionary for different threshold calculations

        # Set the simultaneous threshold to the values for the valid interval if there is only one interval
        if np.sum(ssn_data.vldIntr) == 1:
            ssn_data.bTh = bThI[ssn_data.vldIntr][0]
            ssn_data.mDOO = mDDT  # Metrics dictionary for common threshold, but only valid intervals

            # Determine which threshold to use
            Th = ssn_data.bThI[ssn_data.vldIntr][0]

            # Calculating number of groups in reference data for given threshold
            grpsREFw = np.nansum(np.greater(ssn_data.REF_Dat.values[:, 3:ssn_data.REF_Dat.values.shape[1] - 2], Th),
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

            # Metrics dictionary for common threshold
            mD = {'rSq': np.nan,
                    'mRes': np.nan,
                    'mRRes': np.nan,
                    'Slope': np.nan}

            if obsRefOvrlp:
                mD = self.Calculate_R2M_MRes_MRRes(grpsObsw, grpsREFw, ssn_data.centers, ssn_data.edges, medianSw=ssn_data.medianSw)

            ssn_data.mD = mD  # Metrics dictionary for common threshold
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

        while True:
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

    def GRPsimultaneousEMD(self,
                           nBest1m=100,
                           nBest2m=10):

        """
        Function that peforms the EMD optimization by allowing variations of shift while keeping thresholds constant
        VARIABLES APPENDED TO THE OBJECT ARE SPECIFIED AT THE END

        :param nBest1m: Number of best matches to keep based on optimal distance
        :param nBest2m: Number of best matches to keep based on similarity in number of observations
        :return plot_EMD_obs: whether to plot or not the figures
        """

        ssn_data = self.ssn_data

        print('Optimize EMD by varying shifts, but using the same threshold...', flush=True)

        # Allocating variable to store top matches
        bestThM = np.ones((ssn_data.cenPoints['OBS'].shape[0] + 4 + 2*(ssn_data.EMDbins.shape[0]-1), nBest1m)) * 10000

        # Going through different thresholds for a given combination of shifts
        for TIdx, Thr in enumerate(ssn_data.Thresholds):


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

                # Process only if period is valid
                if ssn_data.vldIntr[siInx]:

                    # Extracting distances and cadences for each threshold
                    x = ssn_data.REF_Grp['FRACYEAR'].values[cadMaskI]
                    yGrp = ssn_data.EMDD[siInx][TIdx, :]

                    # Best index match for each cycle
                    bestCycIn = np.array([])

                    # Finding best shift for each cycle
                    for ceni, cen in enumerate(ssn_data.cenPoints['REF']):

                        # Add index if there is no overlap and is the right kind of interval
                        ObsOvr = np.logical_and(ssn_data.cenPoints['OBS'][siInx][0] >= ssn_data.endPoints['REF'][ceni, 0],
                                                ssn_data.cenPoints['OBS'][siInx][0] <= ssn_data.endPoints['REF'][ceni + 1, 0])
                        if (cen[1] == ssn_data.cenPoints['OBS'][siInx, 1]) and ((not ObsOvr) or (not ssn_data.noOvrlpSw)):
                            bestEMD = np.min(yGrp[np.logical_and(x >= cen[0] - ssn_data.phTol, x <= cen[0] + ssn_data.phTol)])
                            bestin = np.logical_and(yGrp == bestEMD,
                                                    np.logical_and(x >= cen[0] - ssn_data.phTol, x <= cen[0] + ssn_data.phTol)).nonzero()[
                                0][0]
                            bestCycIn = np.append(bestCycIn, bestin)

                    bestCycIn = bestCycIn.astype(int)
                    bestCycIn = np.unique(bestCycIn)

                    # Appending valid indices to variable and storing length
                    valShfInx.append(bestCycIn.astype(int))
                    valShfLen.append(bestCycIn.shape[0])

                # If period is not valid append ones so that they don't add to the permutations
                else:
                    valShfInx.append(1)
                    valShfLen.append(1)

            # Saving lengths as array
            valShfLen = np.array(valShfLen)

            print('Threshold ', Thr, ' - Number of valid combinations:', np.nanprod(valShfLen))
            print(valShfLen)

            for comb in self._mrange(valShfLen):


                # Distribution of daily group counts
                GrpObsI = np.zeros((ssn_data.EMDbins.shape[0]-1))
                GrpREFI = np.zeros((ssn_data.EMDbins.shape[0]-1))

                # Number of active days
                NGrpObs = 0
                NGrpREF = 0

                # Average relative difference in active days between reference and observer
                avDifN = 0

                # Joining ADF from all sub-interval for the specified shifts
                nval = 0
                for siInx in range(0, ssn_data.cenPoints['OBS'].shape[0]):

                    # Append only if period is valid
                    if ssn_data.vldIntr[siInx]:
                        nval = nval + 1

                        GrpObsI += ssn_data.grpDirObsI[siInx][TIdx, valShfInx[siInx][comb[siInx]], :]
                        GrpREFI += ssn_data.grpDirREFI[siInx][TIdx, valShfInx[siInx][comb[siInx]], :]

                        NGrpObs += ssn_data.grpNObsI[siInx][TIdx, valShfInx[siInx][comb[siInx]]]
                        NGrpREF += ssn_data.grpNREFI[siInx][TIdx, valShfInx[siInx][comb[siInx]]]
                        RelDiff = np.abs(ssn_data.grpNObsI[siInx][TIdx, valShfInx[siInx][comb[siInx]]] - ssn_data.grpNREFI[siInx][
                            TIdx, valShfInx[siInx][comb[siInx]]]) / ssn_data.grpNObsI[siInx][TIdx, valShfInx[siInx][comb[siInx]]]
                        #                 maxDifN = np.max([maxDifN, RelDiff])
                        avDifN += RelDiff

                GrpObsI /= nval
                GrpREFI /= nval
                avDifN /= nval

                # Calclulate EMD
                if ssn_data.emdSw:
                    tmpEMD = emd(GrpREFI.astype(np.float64), GrpObsI.astype(np.float64), ssn_data.Dis.astype(np.float64))
                else:
                    tmpEMD = np.sqrt(np.mean(np.power(GrpObsI - GrpREFI, 2)))

                # Insert the combination of intervals if better
                if np.any(bestThM[1, :] > tmpEMD) and np.isfinite(tmpEMD) and tmpEMD:

                    # Initializing array to be inserted
                    insArr = [Thr, tmpEMD, np.abs(NGrpObs-NGrpREF)/NGrpObs, avDifN]

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

                    # Append Distributions for plotting
                    insArr = np.append(insArr, GrpObsI)
                    insArr = np.append(insArr, GrpREFI)

                    # Determining index for insertion
                    insInx = nBest1m - np.sum(bestThM[1, :] >= tmpEMD)

                    # Insert values
                    bestThM = np.insert(bestThM, insInx, insArr, axis=1)

                    # Remove last element
                    bestThM = bestThM[:, 0:nBest1m]


        print('done.', flush=True)
        print(' ', flush=True)

        # Finding best multiple threshold
        bestThM2 = bestThM.copy()

        # Sort according to the average relative difference for all intervals and trimming
        I = np.argsort(bestThM2[3, :], axis=0)
        bestThM2 = np.squeeze(bestThM2[:, I])
        bestThM2 = bestThM2[:, 0:nBest2m]

        # Sort again according to distribution distance
        I = np.argsort(bestThM2[1, :], axis=0)
        bestThM2 = np.squeeze(bestThM2[:, I])

        # Storing best threshold
        bTh = bestThM2[0, 0]

        print('done.', flush=True)
        print(' ', flush=True)

        # Metrics dictionary for common threshold
        mD = {'rSq': np.nan,
              'mRes': np.nan,
              'mRRes': np.nan,
              'Slope': np.nan}

        # Common threshold, but only the valid intervals
        mDOO = {'rSq': np.nan,
                'mRes': np.nan,
                'mRRes': np.nan,
                'Slope': np.nan}

        print('Calculating r-square if there is overlap between observer and reference...', end="", flush=True)

        if ((np.min(ssn_data.REF_Dat['ORDINAL']) <= np.max(ssn_data.ObsDat['ORDINAL'])) and (
                np.max(ssn_data.REF_Dat['ORDINAL']) >= np.max(ssn_data.ObsDat['ORDINAL']))) or (
                (np.max(ssn_data.REF_Dat['ORDINAL']) >= np.min(ssn_data.ObsDat['ORDINAL'])) and (
                np.min(ssn_data.REF_Dat['ORDINAL']) <= np.min(ssn_data.ObsDat['ORDINAL']))):

            # Calculating number of groups in reference data for given threshold
            grpsREFw = np.nansum(np.greater(ssn_data.REF_Dat.values[:, 3:ssn_data.REF_Dat.values.shape[1] - 2], bTh),
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
            mD = self.Calculate_R2M_MRes_MRRes(grpsObsw, grpsREFw, ssn_data.centers, ssn_data.edges, medianSw=ssn_data.medianSw)

            # Calculate R^2 and residual using only valid periods
            calRefN = np.array([0])
            calObsN = np.array([0])
            for n in range(0, ssn_data.cenPoints['OBS'].shape[0]):

                # Perform analysis only if the period is valid and has overlap
                if ssn_data.vldIntr[n] and np.sum(
                        np.logical_and(ssn_data.REF_Dat['FRACYEAR'] >= ssn_data.endPoints['OBS'][n, 0],
                                       ssn_data.REF_Dat['FRACYEAR'] < ssn_data.endPoints['OBS'][
                                           n + 1, 0])) > 0:
                    # Calculating number of groups in reference data for given threshold
                    grpsREFw = np.nansum(
                        np.greater(ssn_data.REF_Dat.values[:, 3:ssn_data.REF_Dat.values.shape[1] - 2], bTh),
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
            mDOO = self.Calculate_R2M_MRes_MRRes(grpsObsw, grpsREFw, ssn_data.centers, ssn_data.edges, medianSw=ssn_data.medianSw)

        print('done.', flush=True)
        print(' ', flush=True)

        # Storing variables in object-----------------------------------------------------------------------------------
        ssn_data.nBest1m = nBest1m  # Number of best matches to keep based on optimal distance
        ssn_data.nBest2m = nBest2m  # Number of best matches to keep based on similarity in number of observations

        ssn_data.bestThM = bestThM  # Variable storing best simultaneous fits
        ssn_data.bestThM2 = bestThM2  # Variable storing best simultaneous fits after trimming using relative active day fraction difference

        ssn_data.bTh = bTh  # Weighted threshold average based on the nBest matches for all simultaneous fits

        ssn_data.mD = mD  # metrics dictionary for common threshold
        ssn_data.mDOO = mDOO  # metrics dictionary for common threshold, but only the valid intervals
        # --------------------------------------------------------------------------------------------------------------

        print('done.', flush=True)
        print(' ', flush=True)

        return True


    def smoothedComparison(self,
                           gssnKrnl=75):

        """
        Function that calculates the smoothed series if there is overlap so that it can be saved in the CSV, if not, returns NaNs
        VARIABLES APPENDED TO THE OBJECT ARE SPECIFIED AT THE END

        :param gssnKrnl:  Width of the gaussian smoothing kernel in days
        """

        ssn_data = self.ssn_data

        # Initializing variables for appending at the end if there is no overlap
        Grp_Comp = []
        mreSth = np.nan
        mneSth = np.nan
        mreMth = np.nan
        mneMth = np.nan
        slpSth = np.nan
        slpMth = np.nan


        if ((np.min(ssn_data.REF_Dat['ORDINAL']) <= np.max(ssn_data.ObsDat['ORDINAL'])) and (
                np.max(ssn_data.REF_Dat['ORDINAL']) >= np.max(ssn_data.ObsDat['ORDINAL']))) or (
                (np.max(ssn_data.REF_Dat['ORDINAL']) >= np.min(ssn_data.ObsDat['ORDINAL'])) and (
                np.min(ssn_data.REF_Dat['ORDINAL']) <= np.min(ssn_data.ObsDat['ORDINAL']))):

            # Creating variables for plotting and calculating difference
            Grp_Comp = ssn_data.REF_Dat[['FRACYEAR', 'ORDINAL', 'YEAR', 'MONTH', 'DAY']].copy()

            # Raw Ref Groups
            Grp_Comp['GROUPS'] = np.nansum(
                np.greater(ssn_data.REF_Dat.values[:, 3:ssn_data.REF_Dat.values.shape[1] - 2], 0), axis=1)
            Grp_Comp['GROUPS'] = Grp_Comp['GROUPS'].astype(float)

            # Thresholded Ref Groups
            Grp_Comp['SINGLETH'] = np.nansum(
                np.greater(ssn_data.REF_Dat.values[:, 3:ssn_data.REF_Dat.values.shape[1] - 2], ssn_data.bTh),
                axis=1).astype(float)
            Grp_Comp['SINGLETHVI'] = Grp_Comp['SINGLETH']

            # Multi-Threshold Ref Groups
            Grp_Comp['MULTITH'] = Grp_Comp['SINGLETH'] * np.nan
            for n in range(0, ssn_data.cenPoints['OBS'].shape[0]):

                # Plot only if the period is valid and has overlap
                if ssn_data.vldIntr[n] and np.sum(
                        np.logical_and(ssn_data.REF_Dat['FRACYEAR'] >= ssn_data.endPoints['OBS'][n, 0],
                                       ssn_data.REF_Dat['FRACYEAR'] < ssn_data.endPoints['OBS'][n + 1, 0])) > 0:
                    intervalmsk = np.logical_and(Grp_Comp['FRACYEAR'] >= ssn_data.endPoints['OBS'][n, 0],
                                                 Grp_Comp['FRACYEAR'] < ssn_data.endPoints['OBS'][n + 1, 0])
                    Grp_Comp.loc[intervalmsk, 'MULTITH'] = np.nansum(
                        np.greater(ssn_data.REF_Dat.values[intervalmsk, 3:ssn_data.REF_Dat.values.shape[1] - 2],
                                   ssn_data.bThI[n]), axis=1).astype(float)

            # Calibrated Observer
            Grp_Comp['CALOBS'] = Grp_Comp['SINGLETH'] * np.nan
            Grp_Comp.loc[np.in1d(ssn_data.REF_Dat['ORDINAL'].values, ssn_data.ObsDat['ORDINAL'].values), 'CALOBS'] = \
                ssn_data.ObsDat.loc[
                    np.in1d(ssn_data.ObsDat['ORDINAL'].values, ssn_data.REF_Dat['ORDINAL'].values), 'GROUPS'].values

            # Imprinting Calibrated Observer NaNs
            nanmsk = np.isnan(Grp_Comp['CALOBS'])
            Grp_Comp.loc[
                np.logical_and(np.in1d(ssn_data.REF_Dat['ORDINAL'].values, ssn_data.ObsDat['ORDINAL'].values),
                               nanmsk), ['CALOBS', 'SINGLETH',
                                         'MULTITH']] = np.nan

            # Imprinting Reference NaNs
            Grp_Comp.loc[np.isnan(ssn_data.REF_Dat['AREA1']), ['CALOBS', 'SINGLETH', 'MULTITH']] = np.nan

            # Adding a Calibrated observer only in valid intervals
            Grp_Comp['CALOBSVI'] = Grp_Comp['CALOBS']
            Grp_Comp.loc[np.isnan(Grp_Comp['MULTITH']), 'CALOBSVI'] = np.nan

            Grp_Comp.loc[np.isnan(Grp_Comp['CALOBS']), 'SINGLETHVI'] = np.nan

            # Smoothing for plotting
            Gss_1D_ker = conv.Gaussian1DKernel(gssnKrnl)
            Grp_Comp['GROUPS'] = conv.convolve(Grp_Comp['GROUPS'].values, Gss_1D_ker, preserve_nan=True)
            Grp_Comp['SINGLETH'] = conv.convolve(Grp_Comp['SINGLETH'].values, Gss_1D_ker, preserve_nan=True)
            Grp_Comp['SINGLETHVI'] = conv.convolve(Grp_Comp['SINGLETHVI'].values, Gss_1D_ker, preserve_nan=True)
            Grp_Comp['MULTITH'] = conv.convolve(Grp_Comp['MULTITH'].values, Gss_1D_ker, preserve_nan=True)
            Grp_Comp['CALOBS'] = conv.convolve(Grp_Comp['CALOBS'].values, Gss_1D_ker, preserve_nan=True)
            Grp_Comp['CALOBSVI'] = conv.convolve(Grp_Comp['CALOBSVI'].values, Gss_1D_ker, preserve_nan=True)
            Grp_Comp['SINGLETHreal'] = Grp_Comp['CALOBSVI']*np.nan

            # Calculate mean normalized error - single threshold
            mreSth = np.round(np.nanmean(np.divide(Grp_Comp['SINGLETHVI'] - Grp_Comp['CALOBS'], Grp_Comp['CALOBS'])),
                              decimals=2)
            mneSth = np.round(np.nanmean(Grp_Comp['SINGLETHVI'] - Grp_Comp['CALOBS']) / np.nanmean(Grp_Comp['CALOBS']),
                              decimals=2)
            slpSth = np.round(np.nanmean(Grp_Comp['SINGLETHVI'] / Grp_Comp['CALOBS']), decimals=2)

            # Calculate mean normalized error - multi threshold
            mreMth = np.round(np.nanmean(np.divide(Grp_Comp['MULTITH'] - Grp_Comp['CALOBSVI'], Grp_Comp['CALOBSVI'])),
                              decimals=2)
            mneMth = np.round(np.nanmean(Grp_Comp['MULTITH'] - Grp_Comp['CALOBSVI']) / np.nanmean(Grp_Comp['CALOBSVI']),
                              decimals=2)
            slpMth = np.round(np.nanmean(Grp_Comp['MULTITH'] / Grp_Comp['CALOBSVI']), decimals=2)

            Grp_Comp['SINGLETHreal'] = np.nansum(
                np.greater(ssn_data.REF_Dat.values[:, 3:ssn_data.REF_Dat.values.shape[1] - 2], ssn_data.optimThS['Optimal']),
                axis=1).astype(float)

            Grp_Comp['SINGLETHreal'] = conv.convolve(Grp_Comp['SINGLETHreal'].values, Gss_1D_ker, preserve_nan=True)

        # Storing variables in object-----------------------------------------------------------------------------------
        ssn_data.Grp_Comp = Grp_Comp  # Smoothed reference and observer series
        ssn_data.mreSth = mreSth  # Mean normalized error - single threshold
        ssn_data.mneSth = mneSth # Mean normalized error with respect to observer group average - single threshold
        ssn_data.slpSth = slpSth  # K-factor between observer and reference for single threshold
        ssn_data.mreMth = mreMth  # Mean normalized error - multi threshold
        ssn_data.mneMth = mneMth # Mean normalized error with respect to observer group average - multi threshold
        ssn_data.slpMth = slpMth  # K-factor between observer and reference for multiple threshold
