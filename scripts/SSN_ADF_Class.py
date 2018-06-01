import pandas as pd
import datetime
import numpy as np
from detect_peaks import detect_peaks
from astropy import convolution as conv
from scipy import signal
import scipy as sp
import matplotlib.pyplot as plt
from matplotlib import colors as clrs
from copy import copy

from pyemd import emd
import os.path

import SSN_Class

class ssnADF_cl(SSN_Class.ssn_cl):

    """
    A class for managing SSN data, reference data, and performing ADF calculations
    """

    def __init__(self,
                 ref_data_path='input_data/SC_SP_RG_DB_KM_group_areas_by_day.csv',
                 silso_path='input_data/SN_m_tot_V2.0.csv',
                 obs_data_path='input_data/GNObservations_JV_V1.22.csv',
                 obs_observer_path='input_data/GNObservers_JV_V1.22.csv',
                 output_path='output',
                 font={'family': 'sans-serif',
                       'weight': 'normal',
                       'size': 21},
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

        SSN_Class.ssn_cl.__init__(self, ref_data_path=ref_data_path,
                                  silso_path=silso_path,
                                  obs_data_path=obs_data_path,
                                  obs_observer_path=obs_observer_path,
                                  output_path=output_path,
                                  font=font)

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

        #--------------------------------------------------------------------------------------------------
        print('Creating window masks...', end="", flush=True)

        risMask = { 'MASK': np.zeros(REF_Dat.shape[0], dtype=bool)}
        decMask = { 'MASK': np.zeros(REF_Dat.shape[0], dtype=bool)}

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

        self.dt = dt  # Temporal Stride in days
        self.phTol = phTol  # Cycle phase tolerance in years
        self.thN = thN  # Number of thresholds including 0
        self.thI = thI  # Threshold increments

        self.risMask = risMask  # Mask indicating the code where to place the search window during raising phases
        self.decMask = decMask  # Mask indicating the code where to place the search window during raising phases

        # --------------------------------------------------------------------------------------------------------------

        if plot:
            self._plotSearchWindows(SILSO_Sn, SIL_max, SIL_min, REF_min, REF_max)

        print('Done initializing data.', flush=True)
        print(' ', flush=True)


    def _plotSearchWindows(self, SILSO_Sn, SIL_max, SIL_min, REF_min, REF_max,
                          dpi=300,
                          pxx=4000,
                          pxy=1300,
                          padv=50,
                          padh=50):

        """
        Function that plots the search window separated for convenience.

        :param SILSO_Sn: Silso sunspot series
        :param SIL_max: Maxima identified in the silso sunspot series
        :param SIL_min: Minima identified in the silso sunspot series
        :param REF_min: Maxima identified in the reference data
        :param REF_max: Minima identified in the reference data

        :param dpi: Dots per inch in figure
        :param pxx: Horizontal size of each panel in pixels
        :param pxy: Vertical size of each panel in pixels
        :param padv: Vertical padding in pixels at the edge of the figure in pixels
        :param padh: Horizontal padding in pixels at the edge of the figure in pixels
        """

        figure_path = self.output_path + '/01_Search_Windows.png'

        print('Creating and saving search window figure...', end="", flush=True)

        font = self.font
        plt.rc('font', **font)

        # Size definitions
        nph = 1  # Number of horizontal panels
        npv = 1  # Number of vertical panels

        # Padding
        padv2 = 0  # Vertical padding in pixels between panels
        padh2 = 0  # Horizontal padding in pixels between panels

        # Figure sizes in pixels
        fszv = (npv * pxy + 2 * padv + (npv - 1) * padv2)  # Vertical size of figure in inches
        fszh = (nph * pxx + 2 * padh + (nph - 1) * padh2)  # Horizontal size of figure in inches

        # Conversion to relative unites
        ppadv = padv / fszv  # Vertical padding in relative units
        ppadv2 = padv2 / fszv  # Vertical padding in relative units
        ppadh = padh / fszv  # Horizontal padding the edge of the figure in relative units
        ppadh2 = padh2 / fszv  # Horizontal padding between panels in relative units

        ## Start Figure
        fig = plt.figure(figsize=(fszh / dpi, fszv / dpi))

        # Average group number
        ax1 = fig.add_axes([ppadh, ppadv, pxx / fszh, pxy / fszv])
        ax1.plot(SILSO_Sn['FRACYEAR'], SILSO_Sn['MMEAN'], color=self.Clr[0], linewidth=2)
        ax1.plot(SILSO_Sn['FRACYEAR'], SILSO_Sn['MSMOOTH'], color=self.Clr[3], linewidth=4)
        ax1.scatter(SIL_max['FRACYEAR'], SIL_max['MSMOOTH'], color='r', edgecolor='r', alpha=1, s=100, linewidths=2, zorder=10)
        ax1.scatter(SIL_min['FRACYEAR'], SIL_min['MSMOOTH'], color='b', edgecolor='b', alpha=1, s=100, linewidths=2, zorder=10)
        ax1.scatter(REF_min['FRACYEAR'], REF_min['MSMOOTH'], color='none', edgecolor='yellow', alpha=1, s=100, linewidths=3, zorder=10)
        ax1.scatter(REF_max['FRACYEAR'], REF_max['MSMOOTH'], color='none', edgecolor='yellow', alpha=1, s=100, linewidths=3, zorder=10)
        ax1.fill(self.REF_Dat['FRACYEAR'], self.risMask['PLOT'] * np.max(SILSO_Sn['MMEAN']), edgecolor=self.Clr[4], color=self.Clr[4], alpha=0.3,
                 zorder=15)
        ax1.fill(self.REF_Dat['FRACYEAR'], self.decMask['PLOT'] * np.max(SILSO_Sn['MMEAN']), edgecolor=self.Clr[2], color=self.Clr[2], alpha=0.3,
                 zorder=15)

        ax1.legend(['Monthly', 'Monthly-Smoothed', 'Search Window (R)', 'Search Window (D)', 'Maxima', 'Minima',
                    'Extrema in Reference'], loc='upper left', ncol=2, frameon=True, edgecolor='none')

        # Axes properties
        ax1.set_xlabel('Year')
        ax1.set_ylabel('Monthly mean total sunspot number')
        ax1.set_xlim(left=np.min(SILSO_Sn['FRACYEAR']), right=np.max(SILSO_Sn['FRACYEAR']))
        ax1.grid(color=(0.5, 0.5, 0.5), linestyle='--', linewidth=1, axis='x', which='both')
        ax1.minorticks_on()
        ax1.set_ylim(bottom=0)

        fig.savefig(figure_path, bbox_inches='tight')

        print('done.', flush=True)
        print(' ', flush=True)


    def processObserver(self,
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

        NamObs = self.GN_Obs['OBSERVER'].values[self.GN_Obs['STATION'].values == CalObs]
        NamObs = NamObs[0]
        NamObs = NamObs[0:NamObs.find(',')].capitalize()

        print('Processing ' + NamObs, flush=True)

        print('Calculating Ordinal day and Fractional year...', end="", flush=True)

        # Picking observations
        ObsDat = self.GN_Dat[self.GN_Dat.STATION == CalObs].copy()

        # If no data for observer exit
        if ObsDat.shape[0] == 0:
            print('done. NO VALID INTERVALS IN OBSERVER', flush=True)
            print(' ', flush=True)
            return False


        ObsDat['ORDINAL'] = ObsDat.apply(lambda x: datetime.date(x['YEAR'].astype(int),x['MONTH'].astype(int),x['DAY'].astype(int)).toordinal(),axis=1)
        ObsDat['FRACYEAR'] = ObsDat.apply(lambda x: x['YEAR'].astype(int)
                                                    + (  datetime.date(x['YEAR'].astype(int),x['MONTH'].astype(int),x['DAY'].astype(int)).toordinal()
                                                       - datetime.date(x['YEAR'].astype(int),1,1).toordinal() )
                                                    / (  datetime.date(x['YEAR'].astype(int)+1,1,1).toordinal()
                                                       - datetime.date(x['YEAR'].astype(int),1,1).toordinal() )
                                          ,axis=1)

        print('done.', flush=True)

        # Finding missing days
        ObsInt = np.arange(np.min(ObsDat['ORDINAL']), np.max(ObsDat['ORDINAL'] + 1))
        MisDays = np.logical_not(sp.in1d(ObsInt, ObsDat['ORDINAL']))

        # Creating dataframe with NaNs for days without observations
        year = np.array(list(map(lambda x: datetime.date.fromordinal(x).year, ObsInt[MisDays])))
        month = np.array(list(map(lambda x: datetime.date.fromordinal(x).month, ObsInt[MisDays])))
        day = np.array(list(map(lambda x: datetime.date.fromordinal(x).day, ObsInt[MisDays])))

        station = day * 0 + CalObs
        observer = day * 0 + ObsDat.iloc[0, 4]
        groups = day * np.nan

        fractyear = np.array(list(map(lambda year, month, day: year + (datetime.date(year, month, day).toordinal()
                                                                       - datetime.date(year, 1, 1).toordinal())
                                                               / (datetime.date(year + 1, 1, 1).toordinal()
                                                                  - datetime.date(year, 1, 1).toordinal()), year, month,
                                      day)))

        NoObs = pd.DataFrame(np.column_stack((year, month, day, station, observer, groups, ObsInt[MisDays], fractyear)),
                             columns=ObsDat.columns.values)

        # Append dataframe with missing days
        ObsDat = ObsDat.append(NoObs, ignore_index=True)

        # Recast using original data types
        origType = ObsDat.dtypes.to_dict()
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
            return False

        grpsOb = grpsOb.reshape((-1, MoLngt))

        # Interval edges for plotting
        fyr1Ob = np.min(yrOb, axis=1)
        fyr2Ob = np.max(yrOb, axis=1)

        obsPlt = { 'X': (fyr1Ob+fyr2Ob)/2}

        # Average number of groups
        Gss_1D_ker = conv.Gaussian1DKernel(2)
        obsPlt['Y'] = conv.convolve(np.nanmean(grpsOb, axis=1), Gss_1D_ker)

        # Finding internal endpoints and centers of Observer Intervals are included if their center is covered by the observer

        # Defining boolean array of valid centers
        validCen = np.logical_and(self.cenPoints['SILSO'][:, 0] > np.min(yrOb), self.cenPoints['SILSO'][:, 0] < np.max(yrOb))


        # Adding a True on the index prior to the first center to include the bracketing point
        validCen[0:validCen.shape[0] - 2] = np.logical_or(validCen[0:validCen.shape[0] - 2],
                                                          validCen[1:validCen.shape[0] - 1])

        # Adding a False at the beggining to account for the difference in size
        validCen = np.insert(validCen, 0, False)

        # Defining arrays
        endPoints = self.endPoints['SILSO'][validCen, :]

        if endPoints.shape[0] == 0:
            endPoints = self.endPoints['SILSO'][0:2, :]
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

        # Storing variables in object-----------------------------------------------------------------------------------

        self.CalObs = CalObs  # Observer identifier denoting observer to be processed
        self.NamObs = NamObs  # Name of observer
        self.minObD = minObD  # Minimum fraction of observed days for an interval to be considered useful
        self.MoLngt = MoLngt  # Duration of the interval ("month") used to calculate the ADF

        self.ObsDat = ObsDat  # Data of observer being analyzed

        self.endPoints['OBS'] = endPoints  # Variable that stores the boundaries of each rising and decaying phase
        self.cenPoints['OBS'] = cenPoints  # Variable that stores the centers of each rising and decaying phase

        self.vldIntr = vldIntr  # Variable indicating whether each rising or decaying interval has enough data to be valid
        self.obsPlt = obsPlt  # Variable with the oserver average groups for plotting


        # --------------------------------------------------------------------------------------------------------------

        # Create folder for observer's output
        if not os.path.exists(self.output_path + '/' + str(CalObs) + '_' + NamObs):
            os.makedirs(self.output_path + '/' + str(CalObs) + '_' + NamObs)

        if np.sum(vldIntr) == 0:
            print('done. NO VALID INTERVALS IN OBSERVER', flush=True)
            print(' ', flush=True)
            return False
        else:
            print('done. Observer has valid intervals', flush=True)
            print(' ', flush=True)

            return True


    def plotActiveVsObserved(self,
                             dpi=300,
                             pxx=4000,
                             pxy=1300,
                             padv=50,
                             padh=50,
                             padv2 = 0,
                             padh2 = 0):

        """

        :param dpi: Dots per inch in figure
        :param pxx: Horizontal size of each panel in pixels
        :param pxy: Vertical size of each panel in pixels
        :param padv: Vertical padding in pixels at the edge of the figure in pixels
        :param padh: Horizontal padding in pixels at the edge of the figure in pixels
        :param padv2: Vertical padding in pixels between panels
        :param padh2: Horizontal padding in pixels between panels
        """

        print('Creating and saving active vs. observed days figure...', end="", flush=True)

        figure_path = self.output_path + '/' + str(self.CalObs) + '_' + self.NamObs + '/02_' + str(self.CalObs) + '_' + self.NamObs + '_active_vs_observed_days.png'

        # Selecting the maximum integer amount of "months" out of the original data
        grpsOb = self.ObsDat['GROUPS'].values
        grpsOb = grpsOb[0:np.int(grpsOb.shape[0] / self.MoLngt) * self.MoLngt]

        ordOb = self.ObsDat['ORDINAL'].values
        ordOb = ordOb[0:np.int(ordOb.shape[0] / self.MoLngt) * self.MoLngt]

        yrOb = self.ObsDat['FRACYEAR'].values
        yrOb = yrOb[0:np.int(yrOb.shape[0] / self.MoLngt) * self.MoLngt]

        # Reshaping
        grpsOb = grpsOb.reshape((-1, self.MoLngt))
        ordOb = ordOb.reshape((-1, self.MoLngt))
        yrOb = yrOb.reshape((-1, self.MoLngt))


        # Number of days with observations
        obsOb = np.sum(np.isfinite(grpsOb), axis=1)

        # Number of days with groups
        grpOb = np.sum(np.greater(grpsOb, 0), axis=1)

        # Average number of groups
        Gss_1D_ker = conv.Gaussian1DKernel(2)
        AvGrpOb = conv.convolve(np.nanmean(grpsOb, axis=1), Gss_1D_ker)
        SdGrpOb = np.nanstd(grpsOb, axis=1)

        # Interval edges for plotting
        fyr1Ob = np.min(yrOb, axis=1)
        fyr2Ob = np.max(yrOb, axis=1)


        # Observer Plot
        # Stack horizontal left ends to level the step-wise plot
        pltxOb = np.stack((fyr1Ob, fyr1Ob)).reshape((1, -1), order='F')

        # Append max fracyear to clapm aria
        pltxOb = np.append(pltxOb, np.max(fyr2Ob))
        pltxOb = np.append(pltxOb, np.max(fyr2Ob))

        # Stack duplicate array to level the step-wise plot
        pltyOb = np.stack((obsOb, obsOb)).reshape((1, -1), order='F')
        pltyGr = np.stack((grpOb, grpOb)).reshape((1, -1), order='F')
        pltyAvOb = np.stack((AvGrpOb, AvGrpOb)).reshape((1, -1), order='F')
        pltySd = np.stack((SdGrpOb, SdGrpOb)).reshape((1, -1), order='F')

        # Append zeros to clamp area
        pltyOb = np.insert(pltyOb, 0, 0)
        pltyOb = np.append(pltyOb, 0)

        pltyGr = np.insert(pltyGr, 0, 0)
        pltyGr = np.append(pltyGr, 0)

        pltyAvOb = np.insert(pltyAvOb, 0, 0)
        pltyAvOb = np.append(pltyAvOb, 0)

        pltySd = np.insert(pltySd, 0, 0)
        pltySd = np.append(pltySd, 0)

        font = self.font
        plt.rc('font', **font)

        # Size definitions
        nph = 1  # Number of horizontal panels
        npv = 3  # Number of vertical panels

        # Figure sizes in pixels
        fszv = (npv * pxy + 2 * padv + (npv - 1) * padv2)  # Vertical size of figure in inches
        fszh = (nph * pxx + 2 * padh + (nph - 1) * padh2)  # Horizontal size of figure in inches

        # Conversion to relative unites
        ppadv = padv / fszv  # Vertical padding in relative units
        ppadv2 = padv2 / fszv  # Vertical padding in relative units
        ppadh = padh / fszv  # Horizontal padding the edge of the figure in relative units
        ppadh2 = padh2 / fszv  # Horizontal padding between panels in relative units

        ## Start Figure
        fig = plt.figure(figsize=(fszh / dpi, fszv / dpi))

        # Dummy axis for right scale
        axd = fig.add_axes([ppadh, ppadv + 2 * pxy / fszv, pxx / fszh, pxy / fszv])
        axd.set_ylim(bottom=0.01, top=1.19)
        axd.set_ylabel('Coverage Fraction')
        axd.yaxis.set_label_position("right")
        axd.yaxis.tick_right()

        # Days with observations and active days
        ax1 = fig.add_axes([ppadh, ppadv + 2 * pxy / fszv, pxx / fszh, pxy / fszv], sharex=axd)
        # Add number of days with observations
        ax1.fill(pltxOb, pltyOb, color=self.Clr[2])
        ax1.fill(pltxOb, pltyGr, color=self.Clr[4])
        # Add number of days with groups (not including zeros and days without observations)

        ax1.plot(np.array([np.min(pltxOb), np.max(pltxOb)]), np.array([1, 1]) * self.minObD * self.MoLngt, 'k--')

        # Axes properties
        ax1.text(0.5, 1.14, 'Comparison of active vs. observed days for ' + self.NamObs,
                 horizontalalignment='center',
                 fontsize=20,
                 transform=ax1.transAxes)
        ax1.set_ylabel('Number of days')
        ax1.legend(['Required Minimum of Observed Days', 'Observed days', 'Active days'], loc='upper right', ncol=3,
                   frameon=True, edgecolor='none')
        ax1.set_xlim(left=np.min(fyr1Ob), right=np.max(fyr2Ob))
        ax1.set_ylim(bottom=0.01 * self.MoLngt, top=1.19 * self.MoLngt)
        ax1.grid(color=(0.5, 0.5, 0.5), linestyle='--', linewidth=1, axis='x', which='both')
        ax1.xaxis.tick_top()
        ax1.minorticks_on()

        # Active/observation/missing mesh
        AcObMesh = np.isfinite(grpsOb).astype(int) + np.greater(grpsOb, 0).astype(int)
        xMesh = np.insert(fyr2Ob, 0, fyr1Ob[0])
        yMesh = np.arange(0, self.MoLngt + 1)

        # Colormap
        colors = [(1, 1, 1), self.Clr[2], self.Clr[4]]
        cmap = clrs.LinearSegmentedColormap.from_list('cmap', colors, N=3)

        ax2 = fig.add_axes([ppadh, ppadv + pxy / fszv, pxx / fszh, pxy / fszv], sharex=axd)
        ax2.pcolormesh(xMesh, yMesh, np.transpose(AcObMesh), cmap=cmap, alpha=0.3, linewidth=2)
        ax2.set_ylim(bottom=0.1, top=self.MoLngt)

        # Axes properties
        ax2.set_ylabel('Day of the month')
        ax2.grid(color=(0.5, 0.5, 0.5), linestyle='--', linewidth=1, axis='x', which='both')
        ax2.minorticks_on()

        # Average group number
        ax3 = fig.add_axes([ppadh, ppadv, pxx / fszh, pxy / fszv])

        for Idx in range(0, self.cenPoints['OBS'].shape[0]):
            if self.vldIntr[Idx]:
                ax3.fill([self.endPoints['OBS'][Idx, 0], self.endPoints['OBS'][Idx, 0], self.endPoints['OBS'][Idx + 1, 0], self.endPoints['OBS'][Idx + 1, 0]],
                         [0, np.ceil(np.nanmax(AvGrpOb)) + 1, np.ceil(np.nanmax(AvGrpOb)) + 1, 0],
                         color=self.Clr[1 + np.mod(Idx, 2) * 2], alpha=0.2)

        ax3.plot((fyr1Ob + fyr2Ob) / 2, AvGrpOb, color=self.Clr[0], linewidth=2)

        # Axes properties
        ax3.set_xlabel('Year')
        ax3.set_ylabel('Average number of groups')
        ax3.set_xlim(left=np.min(fyr1Ob), right=np.max(fyr2Ob))
        ax3.grid(color=(0.5, 0.5, 0.5), linestyle='--', linewidth=1, axis='x', which='both')
        ax3.minorticks_on()
        ax3.set_ylim(bottom=0, top=np.ceil(np.nanmax(AvGrpOb)) + 1);

        fig.savefig(figure_path, bbox_inches='tight')

        print('done.', flush=True)
        print(' ', flush=True)



    def ADFscanningWindowEMD(self,
                             nBest=50):

        """
        Function that preps the search windows and calculates the EMD for each separate rising and decaying interval
        comparing the observer and the reference
        VARIABLES APPENDED TO THE OBJECT ARE SPECIFIED AT THE END

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

        # Going through different sub-intervals
        for siInx in range(0, self.cenPoints['OBS'].shape[0]):

            print('Center:', np.round(self.cenPoints['OBS'][siInx, 0], 2), 'Edges:', np.round(self.endPoints['OBS'][siInx, 0], 2),
                  np.round(self.endPoints['OBS'][siInx + 1, 0], 2))

            # Perform analysis Only if the period is valid
            if self.vldIntr[siInx]:

                print('Valid Interval')

                # Defining mask based on the interval type (rise or decay)
                if self.cenPoints['OBS'][siInx, 1] > 0:
                    cadMaskI = self.risMask['INDEX']
                else:
                    cadMaskI = self.decMask['INDEX']

                # Selecting interval
                TObsDat = self.ObsDat.loc[np.logical_and(self.ObsDat['FRACYEAR'] >= self.endPoints['OBS'][siInx, 0],
                                                         self.ObsDat['FRACYEAR'] < self.endPoints['OBS'][siInx + 1, 0])
                , 'GROUPS'].values.copy()

                TObsFYr = self.ObsDat.loc[np.logical_and(self.ObsDat['FRACYEAR'] >= self.endPoints['OBS'][siInx, 0],
                                                         self.ObsDat['FRACYEAR'] < self.endPoints['OBS'][siInx + 1, 0])
                , 'FRACYEAR'].values.copy()


                # Find index of center of sub-interval
                minYear = np.min(np.absolute(TObsFYr - self.cenPoints['OBS'][siInx, 0]))
                obsMinInx = (np.absolute(TObsFYr - self.cenPoints['OBS'][siInx, 0]) == minYear).nonzero()[0][0]

                # Creating Storing Variables
                # Number of days with groups
                GDObs = np.zeros((self.thN, cadMaskI.shape[0], np.int(TObsDat.shape[0] / self.MoLngt)))
                GDREF = np.zeros((self.thN, cadMaskI.shape[0], np.int(TObsDat.shape[0] / self.MoLngt)))

                # Number of days with observations
                ODObs = np.zeros((self.thN, cadMaskI.shape[0], np.int(TObsDat.shape[0] / self.MoLngt)))
                ODREF = np.zeros((self.thN, cadMaskI.shape[0], np.int(TObsDat.shape[0] / self.MoLngt)))

                # Going through different thresholds
                for TIdx in range(0, self.thN):

                    # Calculating number of groups in reference data for given threshold
                    grpsREFw = np.nansum(np.greater(self.REF_Dat.values[:, 3:self.REF_Dat.values.shape[1] - 3], TIdx * self.thI),
                                         axis=1).astype(float)
                    grpsREFw[np.isnan(self.REF_Dat['AREA1'])] = np.nan

                    # Going through different shifts
                    for SIdx in range(0, cadMaskI.shape[0]):

                        # Selecting the maximum integer amount of "months" out of the original data
                        TgrpsOb = TObsDat[0:np.int(TObsDat.shape[0] / self.MoLngt) * self.MoLngt].copy()

                        # Calculating bracketing indices
                        Idx1 = cadMaskI[SIdx] - obsMinInx
                        Idx2 = Idx1 + TgrpsOb.shape[0]

                        # Selecting reference window of matching size to observer sub-interval;
                        TgrpsREF = grpsREFw[Idx1:Idx2].copy()

                        # Making sure selections have the same length
                        if TgrpsREF.shape[0] == TgrpsOb.shape[0]:
                            # Reshaping into "months"
                            TgrpsOb = TgrpsOb.reshape((-1, self.MoLngt))
                            TgrpsREF = TgrpsREF.reshape((-1, self.MoLngt))

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

            # If period is not valid append empty variavbles
            else:
                print('INVALID Interval')
                GDObs = []
                GDREF = []
                ODObs = []
                ODREF = []

            print(' ')

            # Appending calculated days to list of sub-intervals
            GDObsI.append(GDObs)
            GDREFI.append(GDREF)
            ODObsI.append(ODObs)
            ODREFI.append(ODREF)

        print('done.', flush=True)
        print(' ', flush=True)



        print('Calculating the Earths Mover Distance using a sliding window...', flush=True)

        # Creating Storing dictionaries for distance matrices
        EMDD = []
        EMDtD = []
        EMDthD = []

        # Calculation of distance matrix to be used in the Earth Movers Metric
        x = np.arange(0, self.MoLngt + 1)
        y = np.arange(0, self.MoLngt + 1)
        xx, yy = np.meshgrid(x, y)
        Dis = np.absolute(np.power(xx - yy, 1))

        # Going through different sub-intervals
        for siInx in range(0, self.cenPoints['OBS'].shape[0]):

            print('Center:', np.round(self.cenPoints['OBS'][siInx, 0], 2), 'Edges:', np.round(self.endPoints['OBS'][siInx, 0], 2),
                  np.round(self.endPoints['OBS'][siInx + 1, 0], 2))

            # Perform analysis Only if the period is valid
            if self.vldIntr[siInx]:

                print('Valid Interval')

                # Defining mask based on the interval type (rise or decay)
                if self.cenPoints['OBS'][siInx, 1] > 0:
                    cadMaskI = self.risMask['INDEX']
                else:
                    cadMaskI = self.decMask['INDEX']

                    # Pre-allocating EMD matrix and associated coordinate matrices.  A large default distance valued is used
                # to account for missing points
                EMD = np.ones((GDREFI[siInx].shape[0], GDREFI[siInx].shape[1])) * 1e16
                EMDt = np.zeros((GDREFI[siInx].shape[0], GDREFI[siInx].shape[1]))
                EMDth = np.zeros((GDREFI[siInx].shape[0], GDREFI[siInx].shape[1]))

                # Going through different thresholds
                for TIdx in range(0, self.thN):

                    # Going through different shifts
                    for SIdx in range(0, cadMaskI.shape[0]):

                        if np.any(ODObsI[siInx][TIdx, SIdx, :] != 0) and np.any(ODREFI[siInx][TIdx, SIdx, :] != 0):
                            # Calculating Earth Mover's Distance

                            ADFObs, bins = np.histogram(np.divide(
                                GDObsI[siInx][TIdx, SIdx, ODObsI[siInx][TIdx, SIdx, :] / self.MoLngt >= self.minObD],
                                ODObsI[siInx][TIdx, SIdx, ODObsI[siInx][TIdx, SIdx, :] / self.MoLngt >= self.minObD]),
                                bins=(np.arange(0, self.MoLngt + 2) - 0.5) / self.MoLngt, density=True)

                            ADFREF, bins = np.histogram(np.divide(
                                GDREFI[siInx][TIdx, SIdx, ODREFI[siInx][TIdx, SIdx, :] / self.MoLngt >= self.minObD],
                                ODREFI[siInx][TIdx, SIdx, ODREFI[siInx][TIdx, SIdx, :] / self.MoLngt >= self.minObD]),
                                bins=(np.arange(0, self.MoLngt + 2) - 0.5) / self.MoLngt, density=True)


                            EMD[TIdx, SIdx] = emd(ADFREF.astype(np.float64), ADFObs.astype(np.float64),
                                                  Dis.astype(np.float64))

                        #                 #Calculating Chi-Square distance
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
                        EMDt[TIdx, SIdx] = self.REF_Dat['FRACYEAR'].values[cadMaskI[SIdx]]
                        EMDth[TIdx, SIdx] = TIdx * self.thI

            # If period is not valid append empty variables
            else:
                print('INVALID Interval')
                EMD = []
                EMDt = []
                EMDth = []

            print(' ')

            EMDD.append(EMD)
            EMDtD.append(EMDt)
            EMDthD.append(EMDth)

        print('done.', flush=True)
        print(' ', flush=True)


        print('Identifying the best matches for each valid period and looking for ref-obs overlap...', end="", flush=True)

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
        wAvI = self.vldIntr.copy() * 0
        wSDI = wAvI.copy()


        for siInx in range(0, self.cenPoints['OBS'].shape[0]):

            # Analyze period only if valid
            if self.vldIntr[siInx]:

                # Creating matrix for sorting and find the best combinations of threshold and shift
                OpMat = np.concatenate(
                    (EMDtD[siInx].reshape((-1, 1)), EMDthD[siInx].reshape((-1, 1)),
                     EMDD[siInx].reshape((-1, 1))),
                    axis=1)

                # Sort according to EMD to find the best matches
                I = np.argsort(OpMat[:, 2], axis=0)
                OpMat = np.squeeze(OpMat[I, :])

                # Adding best points
                bestTh.append(OpMat[0:nBest-1, :])

                # Constructing weights
                alph = 1 - (bestTh[siInx][:, 2] - np.min(bestTh[siInx][:, 2])) / (
                        np.max(bestTh[siInx][:, 2]) - np.min(bestTh[siInx][:, 2]))

                if np.isnan(np.sum(alph)):
                    alph = bestTh[siInx][:, 2]*0+1

                # Weighted average
                wAvI[siInx] = np.sum(np.multiply(alph, bestTh[siInx][:, 1])) / np.sum(alph)

                # Weighted Standard Deviation
                wSDI[siInx] = np.sqrt(
                    np.sum(np.multiply(alph, np.power(bestTh[siInx][:, 1] - wAvI[siInx], 2))) / np.sum(alph))

                if np.sum(np.logical_and(self.REF_Dat['FRACYEAR'] >= self.endPoints['OBS'][siInx, 0], self.REF_Dat['FRACYEAR'] < self.endPoints['OBS'][siInx + 1, 0])) > 0:

                    # Activate the overlap switch
                    obsRefOvrlp = True

                    # Calculating number of groups in reference data for given threshold
                    grpsREFw = np.nansum(np.greater(self.REF_Dat.values[:, 3:self.REF_Dat.values.shape[1] - 3], wAvI[siInx]),
                                         axis=1).astype(float)
                    grpsREFw[np.isnan(self.REF_Dat['AREA1'])] = np.nan

                    # Selecting observer's interval
                    TObsDat = self.ObsDat.loc[
                        np.logical_and(self.ObsDat['FRACYEAR'] >= self.endPoints['OBS'][siInx, 0], self.ObsDat['FRACYEAR'] < self.endPoints['OBS'][siInx + 1, 0])
                        , 'GROUPS'].values.copy()
                    TObsOrd = self.ObsDat.loc[
                        np.logical_and(self.ObsDat['FRACYEAR'] >= self.endPoints['OBS'][siInx, 0], self.ObsDat['FRACYEAR'] < self.endPoints['OBS'][siInx + 1, 0])
                        , 'ORDINAL'].values.copy()

                    # Selecting the days of overlap with calibrated observer
                    grpsREFw = grpsREFw[np.in1d(self.REF_Dat['ORDINAL'].values, TObsOrd)]
                    grpsObsw = TObsDat[np.in1d(TObsOrd, self.REF_Dat['ORDINAL'].values)]

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

        self.GDObsI = GDObsI  # Variable that stores the number of days with groups of the observer for each interval, threshold, window shift, and window
        self.ODObsI = ODObsI  # Variable that stores the number of days with observations of the observer for each interval, threshold, window shift, and window
        self.GDREFI = GDREFI  # Variable that stores the number of days with groups of the reference for each interval, threshold, window shift, and window
        self.ODREFI = ODREFI  # Variable that stores the number of days with observations of the reference for each interval, threshold, window shift, and window

        self.EMDD = EMDD  # Variable that stores the EMD between the reference and the observer for each interval, threshold, and window shift
        self.EMDtD = EMDtD  # Variable that stores the windowshift matching EMDD for each interval, threshold, and window shift
        self.EMDthD = EMDthD  # Variable that stores the threshold matching EMDD for each interval, threshold, and window shift

        self.Dis = Dis  # Distance matrix used to calcualte the EMD

        self.nBest = nBest  # Number of top best matches to keep
        self.bestTh = bestTh  # Variable that stores the nBest matches for each interval
        self.wAvI = wAvI  # Weighted threshold average based on the nBest matches for different intervals
        self.wSDI = wSDI  # Weighted threshold standard deviation based on the nBest matches for different intervals

        self.calRef = calRef  # Thresholded number of groups for reference that overlap with observer
        self.calObs = calObs  # Number of groups for observer that overlap with reference

        self.rSqI = rSqI  # R square of the y=x line for each separate interval
        self.mResI = mResI # Mean residual of the y=x line for each separate interval
        self.rSqDT = rSqDT  # R square of the y=x line using the average threshold for each interval
        self.mResDT = mResDT  # Mean residual of the y=x line using the average threshold for each interval
        # --------------------------------------------------------------------------------------------------------------

        print('done.', flush=True)
        print(' ', flush=True)

        return obsRefOvrlp


    def plotOptimalThresholdWindow(self,
                             dpi=300,
                             pxx=4000,
                             pxy=1000,
                             padv=50,
                             padh=50,
                             padv2 = 0,
                             padh2 = 0):

        """

        :param dpi: Dots per inch in figure
        :param pxx: Horizontal size of each panel in pixels
        :param pxy: Vertical size of each panel in pixels
        :param padv: Vertical padding in pixels at the edge of the figure in pixels
        :param padh: Horizontal padding in pixels at the edge of the figure in pixels
        :param padv2: Vertical padding in pixels between panels
        :param padh2: Horizontal padding in pixels between panels
        """

        font = self.font
        plt.rc('font', **font)

        figure_path = self.output_path + '/' + str(self.CalObs) + '_' + self.NamObs + '/03_' + str(
            self.CalObs) + '_' + self.NamObs + '_Optimal_Threshold_Window.png'

        print('Creating and saving optimal threshold figure...', end="", flush=True)

        nph = 1  # Number of horizontal panels
        npv = self.cenPoints['OBS'].shape[0] + 1  # Number of vertical panels

        # Figure sizes in pixels
        fszv = (npv * pxy + 2 * padv + (npv - 1) * padv2)  # Vertical size of figure in inches
        fszh = (nph * pxx + 2 * padh + (nph - 1) * padh2)  # Horizontal size of figure in inches

        # Conversion to relative unites
        ppadv = padv / fszv  # Vertical padding in relative units
        ppadv2 = padv2 / fszv  # Vertical padding in relative units
        ppadh = padh / fszv  # Horizontal padding the edge of the figure in relative units
        ppadh2 = padh2 / fszv  # Horizontal padding between panels in relative units

        ## Start Figure
        fig = plt.figure(figsize=(fszh / dpi, fszv / dpi), dpi=dpi)

        # Comparison with RGO
        ax2 = fig.add_axes([ppadh, ppadv, pxx / fszh, pxy / fszv])
        ax2.plot(self.REF_Dat['FRACYEAR'], self.REF_Dat['AVGROUPS'], 'r--', linewidth=2, alpha=1)

        # Plotting Observer
        ax2.plot(self.obsPlt['X'], self.obsPlt['Y'], color=self.Clr[0], linewidth=2)

        # Axes properties
        ax2.set_ylabel('Average Number of Groups')
        ax2.set_xlabel('Center of sliding window (Year)')
        ax2.set_xlim(left=np.min(self.REF_Dat['FRACYEAR']), right=np.max(self.REF_Dat['FRACYEAR']));
        ax2.set_ylim(bottom=0, top=np.max(self.REF_Dat['AVGROUPS']) * 1.1)

        # EMD Pcolor
        plt.viridis()

        # Going through different sub-intervals
        for siInx in range(0, self.cenPoints['OBS'].shape[0]):

            # Creating axis
            ax1 = fig.add_axes([ppadh, ppadv + (siInx + 1) * (pxy / fszv + ppadv2), pxx / fszh, pxy / fszv])

            # Defining mask based on the interval type (rise or decay)
            if self.cenPoints['OBS'][siInx, 1] > 0:
                cadMaskI = self.risMask['INDEX']
                cadMask = self.risMask['PLOT']
            else:
                cadMaskI = self.decMask['INDEX']
                cadMask = self.decMask['PLOT']

            # Selecting interval
            TObsDat = self.ObsDat.loc[
                np.logical_and(self.ObsDat['FRACYEAR'] >= self.endPoints['OBS'][siInx, 0], self.ObsDat['FRACYEAR'] < self.endPoints['OBS'][siInx + 1, 0])
                , 'GROUPS'].values.copy()
            TObsFYr = self.ObsDat.loc[
                np.logical_and(self.ObsDat['FRACYEAR'] >= self.endPoints['OBS'][siInx, 0], self.ObsDat['FRACYEAR'] < self.endPoints['OBS'][siInx + 1, 0])
                , 'FRACYEAR'].values.copy()
            TObsOrd = self.ObsDat.loc[
                np.logical_and(self.ObsDat['FRACYEAR'] >= self.endPoints['OBS'][siInx, 0], self.ObsDat['FRACYEAR'] < self.endPoints['OBS'][siInx + 1, 0])
                , 'ORDINAL'].values.copy()

            # Find index of minimum inside sub-interval
            minYear = np.min(np.absolute(TObsFYr - self.cenPoints['OBS'][siInx, 0]))
            obsMinInx = (np.absolute(TObsFYr - self.cenPoints['OBS'][siInx, 0]) == minYear).nonzero()[0][0]

            # Calculating mesh for plotting
            x = self.REF_Dat['FRACYEAR'].values[cadMaskI]
            y = np.arange(0, self.thN) * self.thI
            xx, yy = np.meshgrid(x, y)

            # Plot Matrix Only if the period is valid
            if self.vldIntr[siInx]:

                # Creating matrix for sorting and find the best combinations of threshold and shift
                OpMat = np.concatenate(
                    (self.EMDtD[siInx].reshape((-1, 1)), self.EMDthD[siInx].reshape((-1, 1)), self.EMDD[siInx].reshape((-1, 1))),
                    axis=1)

                # Sort according to EMD to find the best matches
                I = np.argsort(OpMat[:, 2], axis=0)
                OpMat = np.squeeze(OpMat[I, :])



                # Initialize varialbes to identify the optimum threshold for the period of overlap
                tmpEMD = 1e16
                tmpt = np.nan
                tmpth = np.nan

                # Calculate optimum threshold for real period of overlap if it exists
                # Check if real if interval is present in Observations
                if (TObsFYr[obsMinInx] > np.min(self.REF_Dat['FRACYEAR'])) and (
                        TObsFYr[obsMinInx] < np.max(self.REF_Dat['FRACYEAR'])):

                    # Check if first element is present in reference
                    if np.any(self.REF_Dat['ORDINAL'] == TObsOrd[0]):

                        # Selecting the maximum integer amount of "months" out of the original data
                        TgrpsOb = TObsDat[0:np.int(TObsDat.shape[0] / self.MoLngt) * self.MoLngt].copy()

                        # Calculating bracketing indices
                        Idx1 = (self.REF_Dat['ORDINAL'] == TObsOrd[0]).nonzero()[0][0]
                        Idx2 = Idx1 + TgrpsOb.shape[0]

                        # Going through different thresholds
                        for TIdx in range(0, self.thN):

                            # Calculating number of groups in reference data for given threshold
                            grpsREFw = np.nansum(
                                np.greater(self.REF_Dat.values[:, 3:self.REF_Dat.values.shape[1] - 3], TIdx * self.thI),
                                axis=1).astype(float)
                            grpsREFw[np.isnan(self.REF_Dat['AREA1'])] = np.nan

                            # Selecting the maximum integer amount of "months" out of the original data
                            TgrpsOb = TObsDat[0:np.int(TObsDat.shape[0] / self.MoLngt) * self.MoLngt].copy()

                            # Selecting reference window of matching size to observer sub-interval;
                            TgrpsREF = grpsREFw[Idx1:Idx2].copy()

                            # Reshaping into "months"
                            TgrpsOb = TgrpsOb.reshape((-1, self.MoLngt))
                            TgrpsREF = TgrpsREF.reshape((-1, self.MoLngt))

                            # Imprinting missing days
                            # OBSERVER
                            TgrpsOb[np.isnan(TgrpsREF)] = np.nan
                            # REFERENCE
                            TgrpsREF[np.isnan(TgrpsOb)] = np.nan

                            # Number of days with groups
                            # OBSERVER
                            GDObsT = np.sum(np.greater(TgrpsOb, 0), axis=1)
                            # REFERENCE
                            GDREFT = np.sum(np.greater(TgrpsREF, 0), axis=1)

                            # Number of days with observations
                            # OBSERVER
                            ODObsT = np.sum(np.isfinite(TgrpsOb), axis=1)
                            # REFERENCE
                            ODREFT = np.sum(np.isfinite(TgrpsREF), axis=1)

                            # Calculating Earth Mover's Distance
                            ADFObs, bins = np.histogram(np.divide(
                                GDObsT[ODObsT / self.MoLngt >= self.minObD],
                                ODObsT[ODObsT / self.MoLngt >= self.minObD]),
                                bins=(np.arange(0, self.MoLngt + 2) - 0.5) / self.MoLngt, density=True)

                            ADFREF, bins = np.histogram(np.divide(
                                GDREFT[ODREFT / self.MoLngt >= self.minObD],
                                ODREFT[ODREFT / self.MoLngt >= self.minObD]),
                                bins=(np.arange(0, self.MoLngt + 2) - 0.5) / self.MoLngt, density=True)

                            tmp = emd(ADFREF.astype(np.float64), ADFObs.astype(np.float64), self.Dis.astype(np.float64))

                            # Calculating Chi-Square distance
                            #ADFObs, bins = np.histogram(GDObsT[ODObsT / MoLngt >= minObD] / MoLngt,
                            #                            bins=(np.arange(0, MoLngt + 2) - 0.5) / MoLngt)
                            #ADFREF, bins = np.histogram(GDREFT[ODREFT / MoLngt >= minObD] / MoLngt,
                            #                            bins=(np.arange(0, MoLngt + 2) - 0.5) / MoLngt)

                            # Calculating numerator and denominator for Chi-square distance
                            #Nom = np.power(ADFObs - ADFREF, 2)
                            # Den = np.power(ADFObs,2) + np.power(ADFREF,2)
                            #Den = ADFObs + ADFREF

                            # Removing zeros in denominator
                            #Nom = Nom[Den != 0]
                            #Den = Den[Den != 0]

                            # Calculating Chi-square distance
                            #tmp = np.sum(np.divide(Nom, Den))

                            # Udating variables
                            if tmp < tmpEMD:
                                tmpEMD = tmp
                                tmpt = TObsFYr[obsMinInx]
                                tmpth = TIdx * self.thI

                OpMat = np.insert(OpMat, 0, [tmpt, tmpth, tmpEMD], axis=0)

                # Plotting Optimization Matrix
                ax1.pcolormesh(xx, yy, self.EMDD[siInx], alpha=1, linewidth=2, vmin=np.min(self.EMDD[siInx]),
                                      vmax=6 * np.min(self.EMDD[siInx]))

                # True Interval
                ax1.scatter(OpMat[0, 0], OpMat[0, 1], c='r', edgecolors='w', linewidths=2, s=200, zorder=11)

                # Best 5 points
                for i in range(1, 5):
                    ax1.scatter(OpMat[i, 0], OpMat[i, 1], c='w', linewidths=2, s=150, zorder=11, alpha=0.5)
                    ax2.plot(
                        self.obsPlt['X'][np.logical_and(self.obsPlt['X'] >= np.min(TObsFYr), self.obsPlt['X'] < np.max(TObsFYr))] - self.cenPoints['OBS'][siInx, 0] +
                        OpMat[i, 0]
                        , self.obsPlt['Y'][np.logical_and(self.obsPlt['X'] >= np.min(TObsFYr), self.obsPlt['X'] < np.max(TObsFYr))],
                        color=self.Clr[5 - siInx], linewidth=3
                        , alpha=0.2)

                # Best 5-10 points
                for i in range(5, 10):
                    ax1.scatter(OpMat[i, 0], OpMat[i, 1], c='w', linewidths=2, s=100, zorder=11, alpha=0.5)
                    ax2.plot(
                        self.obsPlt['X'][np.logical_and(self.obsPlt['X'] >= np.min(TObsFYr), self.obsPlt['X'] < np.max(TObsFYr))] - self.cenPoints['OBS'][siInx, 0] +
                        OpMat[i, 0]
                        , self.obsPlt['Y'][np.logical_and(self.obsPlt['X'] >= np.min(TObsFYr), self.obsPlt['X'] < np.max(TObsFYr))],
                        color=self.Clr[5 - siInx], linewidth=3
                        , alpha=0.2)

                # Best 10-15 points
                for i in range(10, 15):
                    ax1.scatter(OpMat[i, 0], OpMat[i, 1], c='w', linewidths=2, s=50, zorder=11, alpha=0.5)

                # Best 15-20 points
                for i in range(15, 100):
                    ax1.scatter(OpMat[i, 0], OpMat[i, 1], c='w', linewidths=2, s=1, zorder=11, alpha=0.5)

                # Masking Gaps
                pltMsk = np.logical_not(cadMask)
                ax1.fill_between(self.REF_Dat['FRACYEAR'], self.REF_Dat['FRACYEAR'] * 0, y2=self.REF_Dat['FRACYEAR'] * 0 + self.thN,
                                 where=pltMsk, color='w', zorder=10)

                # Plotting real location
                ax1.plot(np.array([1, 1]) * TObsFYr[obsMinInx], np.array([0, np.max(y)]), 'w--', linewidth=3)

            # Plotting edges
            ax1.plot(np.array([1, 1]) * np.min(TObsFYr), np.array([0, np.max(y)]), ':', zorder=11, linewidth=3,
                     color=self.Clr[5 - siInx])
            ax1.plot(np.array([1, 1]) * np.max(TObsFYr), np.array([0, np.max(y)]), ':', zorder=11, linewidth=3,
                     color=self.Clr[5 - siInx])

            ax2.plot(np.array([1, 1]) * np.min(TObsFYr), np.array([0, np.max(y)]), ':', zorder=11, linewidth=3,
                     color=self.Clr[5 - siInx])
            ax2.plot(np.array([1, 1]) * np.max(TObsFYr), np.array([0, np.max(y)]), '-', zorder=11, linewidth=1,
                     color=self.Clr[5 - siInx])
            ax2.plot(np.array([1, 1]) * np.max(TObsFYr), np.array([0, np.max(y)]), ':', zorder=11, linewidth=3,
                     color=self.Clr[5 - siInx])

            # Axes properties
            ax1.set_ylabel('Area threshold (uHem)')
            ax1.set_xlim(left=np.min(self.REF_Dat['FRACYEAR']), right=np.max(self.REF_Dat['FRACYEAR']))
            ax1.set_ylim(bottom=0, top=np.max(y))

            ax1.spines['bottom'].set_color(self.Clr[5 - siInx])
            ax1.spines['bottom'].set_linewidth(3)
            ax1.spines['top'].set_color(self.Clr[5 - siInx])
            ax1.spines['top'].set_linewidth(3)
            ax1.spines['right'].set_color(self.Clr[5 - siInx])
            ax1.spines['right'].set_linewidth(3)
            ax1.spines['left'].set_color(self.Clr[5 - siInx])
            ax1.spines['left'].set_linewidth(3)

            # Adding title
            if siInx == self.cenPoints['OBS'].shape[0] - 1:
                # ax1.text(0.5, 1.01,'Chi-Square (y-y_exp)^2/(y^2+y_exp^2) for ' + NamObs.capitalize(), horizontalalignment='center', transform = ax1.transAxes)
                ax1.text(0.5, 1.01, 'EMD linear distance for ' + self.NamObs, horizontalalignment='center',
                         transform=ax1.transAxes)

        fig.savefig(figure_path, bbox_inches='tight')

        print('done.', flush=True)
        print(' ', flush=True)


    def plotDistributionOfThresholdsMI(self,
                             dpi=300,
                             pxx=1500,
                             pxy=1500,
                             padv=50,
                             padh=50,
                             padv2 = 100,
                             padh2 = 100):

        """

        :param dpi: Dots per inch in figure
        :param pxx: Horizontal size of each panel in pixels
        :param pxy: Vertical size of each panel in pixels
        :param padv: Vertical padding in pixels at the edge of the figure in pixels
        :param padh: Horizontal padding in pixels at the edge of the figure in pixels
        :param padv2: Vertical padding in pixels between panels
        :param padh2: Horizontal padding in pixels between panels
        """

        font = self.font
        plt.rc('font', **font)

        print('Creating and saving distribution of thresholds for different intervals figure...', end="", flush=True)

        figure_path = self.output_path + '/' + str(self.CalObs) + '_' + self.NamObs + '/04_' + str(
            self.CalObs) + '_' + self.NamObs + '_Distribution_of_Thresholds_MI.png'

        frc = 0.8  # Fraction of the panel devoted to histograms

        nph = 3  # Number of horizontal panels
        npv = int(np.ceil(self.vldIntr.shape[0] / nph))  # Number of vertical panels

        # Figure sizes in pixels
        fszv = (npv * pxy + 2 * padv + (npv - 1) * padv2)  # Vertical size of figure in inches
        fszh = (nph * pxx + 2 * padh + (nph - 1) * padh2)  # Horizontal size of figure in inches

        # Conversion to relative unites
        ppadv = padv / fszv  # Vertical padding in relative units
        ppadv2 = padv2 / fszv  # Vertical padding in relative units
        ppadh = padh / fszv  # Horizontal padding the edge of the figure in relative units
        ppadh2 = padh2 / fszv  # Horizontal padding between panels in relative units

        ## Start Figure
        fig = plt.figure(figsize=(fszh / dpi, fszv / dpi), dpi=dpi)
        for i in range(0, nph):
            for j in range(0, npv):

                n = (nph * (j) + i)

                # Only add the panel if it exists
                if n < self.vldIntr.shape[0]:

                    # Plot only if the period is valid
                    if self.vldIntr[n]:
                        # Top Distribution
                        axd = fig.add_axes(
                            [ppadh + i * (pxx / fszh + ppadh2), ppadv + j * (pxy / fszv + ppadv2) + pxy / fszv * frc,
                             pxx / fszh * frc, pxy / fszv * (1 - frc)], label='a' + str(n))
                        axd.hist(self.bestTh[n][:, 2], bins=(np.arange(0, self.nBest, 2)) / self.nBest * (
                                    np.ceil(np.max(self.bestTh[n][:, 2])) - np.floor(np.min(self.bestTh[n][:, 2])))
                                                       + np.floor(np.min(self.bestTh[n][:, 2])), color=self.Clr[4], alpha=.6,
                                 density=True);

                        # Axes properties
                        axd.set_xlim(left=np.floor(np.min(self.bestTh[n][:, 2])), right=np.ceil(np.max(self.bestTh[n][:, 2])))
                        axd.set_axis_off()

                        # Right Distribution
                        ax2 = fig.add_axes(
                            [ppadh + i * (pxx / fszh + ppadh2) + pxx / fszh * frc, ppadv + j * (pxy / fszv + ppadv2),
                             pxx / fszh * frc * (1 - frc), pxy / fszv * frc], label='b' + str(n))
                        ax2.hist(self.bestTh[n][:, 1], bins=np.arange(0, self.thN, self.thI*2), color=self.Clr[2], alpha=.6,
                                 orientation='horizontal', density=True);

                        # # Axes properties
                        ax2.set_ylim(bottom=0, top=self.thN * self.thI)
                        ax2.set_axis_off()

                        # Scatter Plot
                        ax1 = fig.add_axes(
                            [ppadh + i * (pxx / fszh + ppadh2), ppadv + j * (pxy / fszv + ppadv2), pxx / fszh * frc,
                             pxy / fszv * frc], sharex=axd, label='b' + str(n))
                        ax1.scatter(self.bestTh[n][:, 2], self.bestTh[n][:, 1], color="0.25", edgecolor="k", alpha=0.1, s=100,
                                    linewidths=2)

                        ax1.plot(np.array([np.floor(np.min(self.bestTh[n][:, 2])), np.ceil(np.max(self.bestTh[n][:, 2]))]),
                                 np.array([1, 1]) * self.wAvI[n], '--'
                                 , color=self.Clr[4], linewidth=3)
                        ax1.plot(np.array([np.floor(np.min(self.bestTh[n][:, 2])), np.ceil(np.max(self.bestTh[n][:, 2]))]),
                                 np.array([1, 1]) * self.wAvI[n] - self.wSDI[n], ':'
                                 , color=self.Clr[4], linewidth=2)
                        ax1.plot(np.array([np.floor(np.min(self.bestTh[n][:, 2])), np.ceil(np.max(self.bestTh[n][:, 2]))]),
                                 np.array([1, 1]) * self.wAvI[n] + self.wSDI[n], ':'
                                 , color=self.Clr[4], linewidth=2)

                        # Axes properties
                        ax1.set_ylabel('Area threshold (uHem)')
                        ax1.set_xlabel('EMD for ' + self.NamObs)
                        ax1.text(0.5, 0.95, 'From ' + str(np.round(self.endPoints['OBS'][n, 0], decimals=2)) + '  to ' + str(
                            np.round(self.endPoints['OBS'][n + 1, 0], decimals=2)), horizontalalignment='center',
                                 verticalalignment='center', transform=ax1.transAxes)
                        ax1.set_xlim(left=np.floor(np.min(self.bestTh[n][:, 2])), right=np.ceil(np.max(self.bestTh[n][:, 2])))
                        ax1.set_ylim(bottom=0, top=self.thN * self.thI);

        fig.savefig(figure_path, bbox_inches='tight')

        print('done.', flush=True)
        print(' ', flush=True)


    def plotIntervalScatterPlots(self,
                             dpi=300,
                             pxx=1500,
                             pxy=1500,
                             padv=50,
                             padh=50,
                             padv2 = 100,
                             padh2 = 100):

        """

        :param dpi: Dots per inch in figure
        :param pxx: Horizontal size of each panel in pixels
        :param pxy: Vertical size of each panel in pixels
        :param padv: Vertical padding in pixels at the edge of the figure in pixels
        :param padh: Horizontal padding in pixels at the edge of the figure in pixels
        :param padv2: Vertical padding in pixels between panels
        :param padh2: Horizontal padding in pixels between panels
        """

        font = self.font
        plt.rc('font', **font)

        print('Creating and saving interval scatter-plots figure...', end="", flush=True)


        figure_path = self.output_path + '/' + str(self.CalObs) + '_' + self.NamObs + '/05_' + str(
            self.CalObs) + '_' + self.NamObs + '_Interval_Scatter_Plots.png'

        frc = 0.8  # Fraction of the panel devoted to histograms

        nph = 3  # Number of horizontal panels
        npv = int(np.ceil(self.vldIntr.shape[0] / nph))  # Number of vertical panels


        # Figure sizes in pixels
        fszv = (npv * pxy + 2 * padv + (npv - 1) * padv2)  # Vertical size of figure in inches
        fszh = (nph * pxx + 2 * padh + (nph - 1) * padh2)  # Horizontal size of figure in inches

        # Conversion to relative unites
        ppadv = padv / fszv  # Vertical padding in relative units
        ppadv2 = padv2 / fszv  # Vertical padding in relative units
        ppadh = padh / fszv  # Horizontal padding the edge of the figure in relative units
        ppadh2 = padh2 / fszv  # Horizontal padding between panels in relative units

        ## Start Figure
        fig = plt.figure(figsize=(fszh / dpi, fszv / dpi), dpi=dpi)
        for i in range(0, nph):
            for j in range(0, npv):

                n = (nph * (j) + i)

                # Only add the panel if it exists
                if n < self.vldIntr.shape[0]:

                    # Plot only if the period is valid and has overlap
                    if self.vldIntr[n] and np.sum(np.logical_and(self.REF_Dat['FRACYEAR'] >= self.endPoints['OBS'][n, 0],
                                                            self.REF_Dat['FRACYEAR'] < self.endPoints['OBS'][n + 1, 0])) > 0:
                        grpsREFw = self.calRef[n]
                        grpsObsw = self.calObs[n]

                        maxN = np.max([np.nanmax(grpsREFw), np.nanmax(grpsObsw)])

                        # Average group number
                        ax1 = fig.add_axes(
                            [ppadh + i * (pxx / fszh + ppadh2), ppadv + j * (pxy / fszv + ppadv2), pxx / fszh * frc,
                             pxy / fszv * frc], label='b' + str(n))
                        ax1.hist2d(grpsObsw, grpsREFw, bins=np.arange(0, np.ceil(maxN) + 1) - 0.5, cmap=plt.cm.magma_r,
                                   cmin=1)
                        ax1.scatter(grpsObsw, grpsREFw, color='k', edgecolor='k', s=10, linewidths=3, zorder=11,
                                    alpha=0.01)
                        ax1.plot(np.array([-0.5, maxN]), np.array([-0.5, maxN]), '--'
                                 , color=self.Clr[4], linewidth=3)

                        ax1.text(0.5, 0.95, 'From ' + str(np.round(self.endPoints['OBS'][n, 0], decimals=2)) + '  to ' + str(
                            np.round(self.endPoints['OBS'][n + 1, 0], decimals=2)), horizontalalignment='center',
                                 verticalalignment='center', transform=ax1.transAxes)
                        ax1.text(0.5, 0.85, '$R^2$ = ' + str(np.round(self.rSqI[n], decimals=2)) + '  Mn.Res. = ' + str(
                            np.round(self.mResI[n], decimals=2)), horizontalalignment='center', verticalalignment='center',
                                 transform=ax1.transAxes)

                        # Axes properties
                        ax1.set_xlabel('Observed GN for ' + self.NamObs)
                        ax1.set_ylabel('Ref. GN with Th. = ' + str(np.round(self.wAvI[n], decimals=2)))

        fig.savefig(figure_path, bbox_inches='tight')

        print('done.', flush=True)
        print(' ', flush=True)


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
        for siInx in range(0, self.cenPoints['OBS'].shape[0]):

            # Defining mask based on the interval type (rise or decay)
            if self.cenPoints['OBS'][siInx, 1] > 0:
                cadMaskI = self.risMask['INDEX']
            else:
                cadMaskI = self.decMask['INDEX']

            # Plot only if period is valid
            if self.vldIntr[siInx]:

                # Calculating minimum distance
                y = np.amin(self.EMDD[siInx], axis=0)

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
            self.disThres = np.nan  # Threshold above which we will ignore timeshifts
            self.EMDComb = np.nan  # Variable storing best simultaneous fits

            self.wAv = np.nan # Weighted threshold average based on the nBest matches for all simultaneous fits
            self.wSD = np.nan  # Weighted threshold standard deviation based on the nBest matches for all simultaneous fits

            self.rSq = np.nan  # R square of the y=x line using a common threshold
            self.mRes = np.nan  # Mean residual of the y=x line using a common threshold

            self.rSqOO = np.nan  # R square of the y=x line using a common threshold, but only the valid intervals
            self.mResOO = np.nan  # Mean residual of the y=x line using a common threshold, but only the valid intervals

            return False



        print('Optimize EMD by varying shifts, but using the same threshold...', flush=True)

        # Allocating variable to store top matches
        EMDComb = np.ones((self.cenPoints['OBS'].shape[0] + 2, self.nBest)) * 10000

        # Identify first valid index
        fstVldIn = self.vldIntr.nonzero()[0][0]

        print('start', datetime.datetime.now(), '0 of', valShfLen[fstVldIn] - 1)

        comProg = 0
        for comb in self._mrange(valShfLen):

            # Inform user of progress
            if comb[fstVldIn] != comProg:
                print(comb[fstVldIn], 'of', valShfLen[fstVldIn] - 1, 'at', datetime.datetime.now())
                comProg = comb[fstVldIn]

                # Going through different thresholds for a given combination of shifts
            for TIdx in range(0, self.thN):

                # Initializing arrays for joining the ADFs of all sub-intervals
                ADFObsI = np.array([])
                ADFREFI = np.array([])

                # Joining ADF from all sub-interval for the specified shifts
                for siInx in range(0, self.cenPoints['OBS'].shape[0]):

                    # Append only if period is valid
                    if self.vldIntr[siInx]:

                        # If it is the first interval re-create the arrays
                        if ADFObsI.shape[0] == 0:

                            ADFObsI = np.divide(
                                self.GDObsI[siInx][TIdx, valShfInx[siInx][comb[siInx]], self.ODObsI[siInx][TIdx, valShfInx[siInx][comb[siInx]], :] / self.MoLngt >= self.minObD],
                                self.ODObsI[siInx][TIdx, valShfInx[siInx][comb[siInx]], self.ODObsI[siInx][TIdx, valShfInx[siInx][comb[siInx]], :] / self.MoLngt >= self.minObD])

                            ADFREFI = np.divide(
                                self.GDREFI[siInx][TIdx, valShfInx[siInx][comb[siInx]], self.ODREFI[siInx][TIdx, valShfInx[siInx][comb[siInx]], :] / self.MoLngt >= self.minObD],
                                self.ODREFI[siInx][TIdx, valShfInx[siInx][comb[siInx]], self.ODREFI[siInx][TIdx, valShfInx[siInx][comb[siInx]], :] / self.MoLngt >= self.minObD])

                        # If not, append ADF from all sub-interval for the specified shifts
                        else:
                            ADFObsI = np.append(ADFObsI, np.divide(
                                self.GDObsI[siInx][TIdx, valShfInx[siInx][comb[siInx]], self.ODObsI[siInx][TIdx, valShfInx[siInx][comb[siInx]], :] / self.MoLngt >= self.minObD],
                                self.ODObsI[siInx][TIdx, valShfInx[siInx][comb[siInx]], self.ODObsI[siInx][TIdx, valShfInx[siInx][comb[siInx]], :] / self.MoLngt >= self.minObD]))

                            ADFREFI = np.append(ADFREFI, np.divide(
                                self.GDREFI[siInx][TIdx, valShfInx[siInx][comb[siInx]], self.ODREFI[siInx][TIdx, valShfInx[siInx][comb[siInx]], :] / self.MoLngt >= self.minObD],
                                self.ODREFI[siInx][TIdx, valShfInx[siInx][comb[siInx]], self.ODREFI[siInx][TIdx, valShfInx[siInx][comb[siInx]], :] / self.MoLngt >= self.minObD]))

                # Calculating Earth Mover's Distance
                ADFObs, bins = np.histogram(ADFObsI, bins=(np.arange(0, self.MoLngt + 2) - 0.5) / self.MoLngt, density=True)
                ADFREF, bins = np.histogram(ADFREFI, bins=(np.arange(0, self.MoLngt + 2) - 0.5) / self.MoLngt, density=True)
                tmpEMD = emd(ADFREF.astype(np.float64), ADFObs.astype(np.float64), self.Dis.astype(np.float64))

                if np.any(EMDComb[0, :] > tmpEMD):

                    # Determining index for insertion
                    insInx = self.nBest - np.sum(EMDComb[0, :] >= tmpEMD)

                    # Initializing array to be inserted
                    insArr = [tmpEMD, TIdx]

                    # Append shifts
                    for siInx in range(0, self.cenPoints['OBS'].shape[0]):

                        # Append only if period is valid
                        if self.vldIntr[siInx]:
                            insArr.append(valShfInx[siInx][comb[siInx]])
                        # If not, append dummy
                        else:
                            insArr.append(np.nan)

                    # Convert to numpy array
                    insArr = np.array(insArr)

                    # Insert values
                    EMDComb = np.insert(EMDComb, insInx, insArr, axis=1)

                    # Remove last element
                    EMDComb = EMDComb[:, 0:self.nBest]

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
        if (np.min(self.REF_Dat['ORDINAL']) <= np.min(self.ObsDat['ORDINAL'])) or (
                np.max(self.REF_Dat['ORDINAL']) >= np.max(self.ObsDat['ORDINAL'])):

            # Calculating number of groups in reference data for given threshold
            grpsREFw = np.nansum(np.greater(self.REF_Dat.values[:, 3:self.REF_Dat.values.shape[1] - 3], wAv), axis=1).astype(float)
            grpsREFw[np.isnan(self.REF_Dat['AREA1'])] = np.nan

            # Selecting the days of overlap with calibrated observer
            grpsREFw = grpsREFw[np.in1d(self.REF_Dat['ORDINAL'].values, self.ObsDat['ORDINAL'].values)]
            grpsObsw = self.ObsDat.loc[np.in1d(self.ObsDat['ORDINAL'].values, self.REF_Dat['ORDINAL'].values), 'GROUPS'].values

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
            for n in range(0, self.cenPoints['OBS'].shape[0]):

                # Plot only if the period is valid and has overlap
                if self.vldIntr[n] and np.sum(np.logical_and(self.REF_Dat['FRACYEAR'] >= self.endPoints['OBS'][n, 0],
                                                        self.REF_Dat['FRACYEAR'] < self.endPoints['OBS'][n + 1, 0])) > 0:
                    # Calculating number of groups in reference data for given threshold
                    grpsREFw = np.nansum(np.greater(self.REF_Dat.values[:, 3:self.REF_Dat.values.shape[1] - 3], wAv),
                                         axis=1).astype(float)
                    grpsREFw[np.isnan(self.REF_Dat['AREA1'])] = np.nan

                    # Selecting observer's interval
                    TObsDat = self.ObsDat.loc[
                        np.logical_and(self.ObsDat['FRACYEAR'] >= self.endPoints['OBS'][n, 0], self.ObsDat['FRACYEAR'] < self.endPoints['OBS'][n + 1, 0])
                        , 'GROUPS'].values.copy()
                    TObsOrd = self.ObsDat.loc[
                        np.logical_and(self.ObsDat['FRACYEAR'] >= self.endPoints['OBS'][n, 0], self.ObsDat['FRACYEAR'] < self.endPoints['OBS'][n + 1, 0])
                        , 'ORDINAL'].values.copy()

                    # Selecting the days of overlap with calibrated observer
                    grpsREFw = grpsREFw[np.in1d(self.REF_Dat['ORDINAL'].values, TObsOrd)]
                    grpsObsw = TObsDat[np.in1d(TObsOrd, self.REF_Dat['ORDINAL'].values)]

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

        self.disThres = disThres  # Threshold above which we will ignore timeshifts
        self.EMDComb = EMDComb  # Variable storing best simultaneous fits

        self.wAv = wAv  # Weighted threshold average based on the nBest matches for all simultaneous fits
        self.wSD = wSD  # Weighted threshold standard deviation based on the nBest matches for all simultaneous fits

        self.rSq  = rSq  # R square of the y=x line using a common threshold
        self.mRes = mRes # Mean residual of the y=x line using a common threshold

        self.rSqOO  = rSqOO  # R square of the y=x line using a common threshold, but only the valid intervals
        self.mResOO = mResOO # Mean residual of the y=x line using a common threshold, but only the valid intervals
        # --------------------------------------------------------------------------------------------------------------

        print('done.', flush=True)
        print(' ', flush=True)

        return True

    def plotMinEMD(self,
                   dpi=300,
                   pxx=4000,
                   pxy=1000,
                   padv=30,
                   padh=30,
                   padv2 = 0,
                   padh2 = 0):

        """

        :param dpi: Dots per inch in figure
        :param pxx: Horizontal size of each panel in pixels
        :param pxy: Vertical size of each panel in pixels
        :param padv: Vertical padding in pixels at the edge of the figure in pixels
        :param padh: Horizontal padding in pixels at the edge of the figure in pixels
        :param padv2: Vertical padding in pixels between panels
        :param padh2: Horizontal padding in pixels between panels
        """

        print('Creating and saving minimum EMD figure...', end="", flush=True)

        figure_path = self.output_path + '/' + str(self.CalObs) + '_' + self.NamObs + '/06_' + str(
            self.CalObs) + '_' + self.NamObs + '_Min_EMD.png'

        font = self.font
        plt.rc('font', **font)

        nph = 1  # Number of horizontal panels
        npv = self.cenPoints['OBS'].shape[0] + 1  # Number of vertical panels

        # Figure sizes in pixels
        fszv = (npv * pxy + 2 * padv + (npv - 1) * padv2)  # Vertical size of figure in inches
        fszh = (nph * pxx + 2 * padh + (nph - 1) * padh2)  # Horizontal size of figure in inches

        # Conversion to relative unites
        ppadv = padv / fszv  # Vertical padding in relative units
        ppadv2 = padv2 / fszv  # Vertical padding in relative units
        ppadh = padh / fszv  # Horizontal padding the edge of the figure in relative units
        ppadh2 = padh2 / fszv  # Horizontal padding between panels in relative units

        ## Start Figure
        fig = plt.figure(figsize=(fszh / dpi, fszv / dpi), dpi=dpi)

        # Comparison with RGO
        ax2 = fig.add_axes([ppadh, ppadv, pxx / fszh, pxy / fszv])
        ax2.plot(self.REF_Dat['FRACYEAR'], self.REF_Dat['AVGROUPS'], 'r--', linewidth=2, alpha=1)

        # Plotting Observer
        ax2.plot(self.obsPlt['X'], self.obsPlt['Y'], color=self.Clr[0], linewidth=2)

        # Axes properties
        ax2.set_ylabel('Average Number of Groups')
        ax2.set_xlabel('Center of sliding window (Year)')
        ax2.set_xlim(left=np.min(self.REF_Dat['FRACYEAR']), right=np.max(self.REF_Dat['FRACYEAR']));
        ax2.set_ylim(bottom=0, top=np.max(self.REF_Dat['AVGROUPS']) * 1.1)

        # Going through different sub-intervals
        for siInx in range(0, self.cenPoints['OBS'].shape[0]):

            # Defining mask based on the interval type (rise or decay)
            if self.cenPoints['OBS'][siInx, 1] > 0:
                cadMaskI = self.risMask['INDEX']
                cadMask = self.risMask['PLOT']
            else:
                cadMaskI = self.decMask['INDEX']
                cadMask = self.decMask['PLOT']

            # Creating axis
            ax1 = fig.add_axes([ppadh, ppadv + (siInx + 1) * (pxy / fszv + ppadv2), pxx / fszh, pxy / fszv])

            TObsFYr = self.ObsDat.loc[
                np.logical_and(self.ObsDat['FRACYEAR'] >= self.endPoints['OBS'][siInx, 0], self.ObsDat['FRACYEAR'] < self.endPoints['OBS'][siInx + 1, 0])
                , 'FRACYEAR'].values.copy()

            # Initializing varialble to plot non-valid intervals
            x = self.REF_Dat['FRACYEAR'].values[cadMaskI]
            y = np.array([1, 10])

            # Plot only if period is valid
            if self.vldIntr[siInx]:

                # Calculating minimum distance for plotting
                y = np.amin(self.EMDD[siInx], axis=0)

                # Plotting Optimization Matrix
                ax1.plot(x, y, color='k', linewidth=3)

                # Masking Gaps
                pltMsk = np.logical_not(cadMask)
                ax1.fill_between(self.REF_Dat['FRACYEAR'], self.REF_Dat['FRACYEAR'] * 0,
                                 y2=self.REF_Dat['FRACYEAR'] * 0 + np.min(y) * 10, where=pltMsk, color='w', zorder=10)

                # Plotting possible theshold
                ax1.plot(np.array([np.min(x), np.max(x)]), np.array([1, 1]) * self.disThres * np.min(y), 'b:', linewidth=3)

            # Plotting edges
            ax1.plot(np.array([1, 1]) * np.min(TObsFYr), np.array([0, np.max(y)]), ':', zorder=11, linewidth=3,
                     color=self.Clr[5 - siInx])
            ax1.plot(np.array([1, 1]) * np.max(TObsFYr), np.array([0, np.max(y)]), ':', zorder=11, linewidth=3,
                     color=self.Clr[5 - siInx])

            ax2.plot(np.array([1, 1]) * np.min(TObsFYr), np.array([0, np.max(y)]), ':', zorder=11, linewidth=3,
                     color=self.Clr[5 - siInx])
            ax2.plot(np.array([1, 1]) * np.max(TObsFYr), np.array([0, np.max(y)]), '-', zorder=11, linewidth=1,
                     color=self.Clr[5 - siInx])
            ax2.plot(np.array([1, 1]) * np.max(TObsFYr), np.array([0, np.max(y)]), ':', zorder=11, linewidth=3,
                     color=self.Clr[5 - siInx])

            # Axes properties
            ax1.set_ylabel('Distribution Distance')
            ax1.set_xlim(left=np.min(self.REF_Dat['FRACYEAR']), right=np.max(self.REF_Dat['FRACYEAR']))
            ax1.set_ylim(bottom=0, top=np.min(y) * 5 + 1)

            ax1.spines['bottom'].set_color(self.Clr[5 - siInx])
            ax1.spines['bottom'].set_linewidth(3)
            ax1.spines['top'].set_color(self.Clr[5 - siInx])
            ax1.spines['top'].set_linewidth(3)
            ax1.spines['right'].set_color(self.Clr[5 - siInx])
            ax1.spines['right'].set_linewidth(3)
            ax1.spines['left'].set_color(self.Clr[5 - siInx])
            ax1.spines['left'].set_linewidth(3)

            # Adding title
            if siInx == self.cenPoints['OBS'].shape[0] - 1:
                # ax1.text(0.5, 1.01,'Chi-Square (y-y_exp)^2/(y^2+y_exp^2) for ' + NamObs.capitalize(), horizontalalignment='center', transform = ax1.transAxes)
                ax1.text(0.5, 1.01, 'EMD linear distance for ' + self.NamObs, horizontalalignment='center',
                         transform=ax1.transAxes)

        fig.savefig(figure_path, bbox_inches='tight')

        print('done.', flush=True)
        print(' ', flush=True)

    def plotSimultaneousFit(self,
                             dpi=300,
                             pxx=4000,
                             pxy=1000,
                             padv=50,
                             padh=50,
                             padv2 = 0,
                             padh2 = 0):

        """

        :param dpi: Dots per inch in figure
        :param pxx: Horizontal size of each panel in pixels
        :param pxy: Vertical size of each panel in pixels
        :param padv: Vertical padding in pixels at the edge of the figure in pixels
        :param padh: Horizontal padding in pixels at the edge of the figure in pixels
        :param padv2: Vertical padding in pixels between panels
        :param padh2: Horizontal padding in pixels between panels
        """

        print('Creating and saving simultaneous fit figure...', end="", flush=True)

        figure_path = self.output_path + '/' + str(self.CalObs) + '_' + self.NamObs + '/07_' + str(
            self.CalObs) + '_' + self.NamObs + '_Simultaneous_Fit.png'

        font = self.font
        plt.rc('font', **font)

        frc = 0.9  # Fraction of the panel devoted to histogram

        nph = 1  # Number of horizontal panels
        npv = 2  # Number of vertical panels

        # Figure sizes in pixels
        fszv = (npv * pxy + 2 * padv + (npv - 1) * padv2)  # Vertical size of figure in inches
        fszh = (nph * pxx + 2 * padh + (nph - 1) * padh2)  # Horizontal size of figure in inches

        # Conversion to relative unites
        ppadv = padv / fszv  # Vertical padding in relative units
        ppadv2 = padv2 / fszv  # Vertical padding in relative units
        ppadh = padh / fszv  # Horizontal padding the edge of the figure in relative units
        ppadh2 = padh2 / fszv  # Horizontal padding between panels in relative units

        ## Start Figure
        fig = plt.figure(figsize=(fszh / dpi, fszv / dpi), dpi=dpi)

        # Comparison with RGO
        ax2 = fig.add_axes([ppadh, ppadv, pxx / fszh * frc, pxy / fszv])
        ax2.plot(self.REF_Dat['FRACYEAR'], self.REF_Dat['AVGROUPS'], 'r--', linewidth=2, alpha=1)

        # Plotting Observer
        ax2.plot(self.obsPlt['X'], self.obsPlt['Y'], color=self.Clr[0], linewidth=2)

        # Axes properties
        ax2.set_ylabel('Av. Num. of Groups')
        ax2.set_xlabel('Center of sliding window (Year)')
        ax2.set_xlim(left=np.min(self.REF_Dat['FRACYEAR']), right=np.max(self.REF_Dat['FRACYEAR']));
        ax2.set_ylim(bottom=0, top=np.max(self.REF_Dat['AVGROUPS']) * 1.1)

        # Placement of top simultaneous fits
        ax1 = fig.add_axes([ppadh, ppadv + (pxy / fszv + ppadv2), pxx / fszh * frc, pxy / fszv])

        # Going through different sub-intervals
        for siInx in range(0, self.cenPoints['OBS'].shape[0]):
            TObsFYr = self.ObsDat.loc[
                np.logical_and(self.ObsDat['FRACYEAR'] >= self.endPoints['OBS'][siInx, 0], self.ObsDat['FRACYEAR'] < self.endPoints['OBS'][siInx + 1, 0])
                , 'FRACYEAR'].values.copy()

            # Plotting edges
            ax1.plot(np.array([1, 1]) * np.min(TObsFYr), np.array([0, self.thN * self.thI]), ':', zorder=11, linewidth=3,
                     color=self.Clr[5 - siInx])
            ax1.plot(np.array([1, 1]) * np.max(TObsFYr), np.array([0, self.thN * self.thI]), '-', zorder=11, linewidth=1,
                     color=self.Clr[5 - siInx])
            ax1.plot(np.array([1, 1]) * np.max(TObsFYr), np.array([0, self.thN * self.thI]), ':', zorder=11, linewidth=3,
                     color=self.Clr[5 - siInx])

            ax2.plot(np.array([1, 1]) * np.min(TObsFYr), np.array([0, self.thN * self.thI]), ':', zorder=11, linewidth=3,
                     color=self.Clr[5 - siInx])
            ax2.plot(np.array([1, 1]) * np.max(TObsFYr), np.array([0, self.thN * self.thI]), '-', zorder=11, linewidth=1,
                     color=self.Clr[5 - siInx])
            ax2.plot(np.array([1, 1]) * np.max(TObsFYr), np.array([0, self.thN * self.thI]), ':', zorder=11, linewidth=3,
                     color=self.Clr[5 - siInx])

        for i in range(0, self.nBest):

            # Initialize plot vector
            x = np.array([])

            for siInx in range(0, self.cenPoints['OBS'].shape[0]):

                # Defining mask based on the interval type (rise or decay)
                if self.cenPoints['OBS'][siInx, 1] > 0:
                    cadMaskI = self.risMask['INDEX']
                else:
                    cadMaskI = self.decMask['INDEX']

                Year = self.REF_Dat['FRACYEAR'].values[cadMaskI]

                # Append only if period is valid
                if self.vldIntr[siInx]:

                    # If it is the first interval re-create the array
                    if x.shape[0] == 0:
                        x = np.array([Year[self.EMDComb[siInx + 2, i].astype(np.int)]])
                    # Append other sub-intervals
                    else:
                        x = np.append(x, Year[self.EMDComb[siInx + 2, i].astype(np.int)])

                        # Creating matching threshold vector
            y = x * 0 + self.EMDComb[1, i]

            # Constructing alpha
            alph = 1 - (self.EMDComb[0, i] - np.min(self.EMDComb[0, :])) / (np.max(self.EMDComb[0, :]) - np.min(self.EMDComb[0, :]))

            # Plotting Intervals
            ax1.plot(x, y, 'o:', zorder=11, linewidth=1, color=self.Clr[2], alpha=alph, markersize=(101 - i) / 8)

        # Axes properties
        ax1.set_ylabel('Area thres. (uHem)')
        ax1.set_xlim(left=np.min(self.REF_Dat['FRACYEAR']), right=np.max(self.REF_Dat['FRACYEAR']))
        ax1.set_ylim(bottom=0, top=self.thN * self.thI)

        ax1.text(0.5, 1.01, 'Best Simultaneous Fits for ' + self.NamObs, horizontalalignment='center',
                 transform=ax1.transAxes);

        # Right Distribution
        ax3 = fig.add_axes(
            [ppadh + pxx / fszh * frc, ppadv + (pxy / fszv + ppadv2), pxx / fszh * (1 - frc), pxy / fszv])
        ax3.hist(self.EMDComb[1, :], bins=np.arange(0, self.thN) * self.thI + self.thI / 2, color=self.Clr[2], alpha=.6,
                 orientation='horizontal', density=True);
        # ax3.plot(yOD, xOD, color=Clr[2], linewidth=3)

        # # Axes properties
        ax3.set_ylim(bottom=0, top=self.thN * self.thI)
        ax3.set_axis_off()

        fig.savefig(figure_path, bbox_inches='tight')

        print('done.', flush=True)
        print(' ', flush=True)


    def plotDistributionOfThresholds(self,
                             dpi=300,
                             pxx=3000,
                             pxy=3000,
                             padv=50,
                             padh=50,
                             padv2=0,
                             padh2=0):

        """

        :param dpi: Dots per inch in figure
        :param pxx: Horizontal size of each panel in pixels
        :param pxy: Vertical size of each panel in pixels
        :param padv: Vertical padding in pixels at the edge of the figure in pixels
        :param padh: Horizontal padding in pixels at the edge of the figure in pixels
        :param padv2: Vertical padding in pixels between panels
        :param padh2: Horizontal padding in pixels between panels
        """

        font = self.font
        plt.rc('font', **font)

        print('Creating and saving distribution of thresholds for different intervals figure...', end="", flush=True)

        figure_path = self.output_path + '/' + str(self.CalObs) + '_' + self.NamObs + '/08_' + str(
            self.CalObs) + '_' + self.NamObs + '_Distribution_of_Thresholds.png'

        # Distribution Plots of threshold and distance
        frc = 0.8  # Fraction of the panel devoted to histograms

        nph = 1  # Number of horizontal panels
        npv = 1  # Number of vertical panels

        # Figure sizes in pixels
        fszv = (npv * pxy + 2 * padv + (npv - 1) * padv2)  # Vertical size of figure in inches
        fszh = (nph * pxx + 2 * padh + (nph - 1) * padh2)  # Horizontal size of figure in inches

        # Conversion to relative unites
        ppadv = padv / fszv  # Vertical padding in relative units
        ppadv2 = padv2 / fszv  # Vertical padding in relative units
        ppadh = padh / fszv  # Horizontal padding the edge of the figure in relative units
        ppadh2 = padh2 / fszv  # Horizontal padding between panels in relative units

        ## Start Figure
        fig = plt.figure(figsize=(fszh / dpi, fszv / dpi))

        # Top Distribution
        axd = fig.add_axes([ppadh, ppadv + pxy / fszv * frc, pxx / fszh * frc, pxy / fszv * (1 - frc)])
        axd.hist(self.EMDComb[0, :],
                 bins=(np.arange(0, self.nBest, 2)) / self.nBest * (
                         np.ceil(np.max(self.EMDComb[0, :])) - np.floor(np.min(self.EMDComb[0, :])))
                      + np.floor(np.min(self.EMDComb[0, :])), color=self.Clr[4], alpha=.6, density=True)

        # Axes properties
        axd.set_xlim(left=np.floor(np.min(self.EMDComb[0, :])), right=np.ceil(np.max(self.EMDComb[0, :])))
        axd.set_axis_off()

        # Right Distribution
        ax2 = fig.add_axes([ppadh + pxx / fszh * frc, ppadv, pxx / fszh * frc * (1 - frc), pxy / fszv * frc])
        ax2.hist(self.EMDComb[1, :], bins=np.arange(0, self.thN, self.thI*2), color=self.Clr[2], alpha=.6,
                                 orientation='horizontal', density=True);
        # ax2.plot(yOD, xOD, color=Clr[2], linewidth=3)

        # # Axes properties
        ax2.set_ylim(bottom=0, top=self.thN * self.thI)
        ax2.set_axis_off()

        # Scatter Plot
        ax1 = fig.add_axes([ppadh, ppadv, pxx / fszh * frc, pxy / fszv * frc], sharex=axd)
        ax1.scatter(self.EMDComb[0, :], self.EMDComb[1, :], color="0.25", edgecolor="k", alpha=0.1, s=100, linewidths=2)

        ax1.plot(np.array([np.floor(np.min(self.EMDComb[0, :])), np.ceil(np.max(self.EMDComb[0, :]))]), np.array([1, 1]) * self.wAv,
                 '--'
                 , color=self.Clr[4], linewidth=3)
        ax1.plot(np.array([np.floor(np.min(self.EMDComb[0, :])), np.ceil(np.max(self.EMDComb[0, :]))]),
                 np.array([1, 1]) * self.wAv - self.wSD, ':'
                 , color=self.Clr[4], linewidth=2)
        ax1.plot(np.array([np.floor(np.min(self.EMDComb[0, :])), np.ceil(np.max(self.EMDComb[0, :]))]),
                 np.array([1, 1]) * self.wAv + self.wSD, ':'
                 , color=self.Clr[4], linewidth=2)

        # Axes properties
        ax1.set_ylabel('Area threshold (uHem)')
        ax1.text(1.02, 0.52, 'Area threshold (uHem)', horizontalalignment='center', transform=ax1.transAxes,
                 rotation='vertical', verticalalignment='center')
        ax1.set_xlabel('EMD')
        ax1.text(0.5, 1.01, 'EMD for ' + self.NamObs, horizontalalignment='center', transform=ax1.transAxes)
        ax1.set_xlim(left=np.floor(np.min(self.EMDComb[0, :])), right=np.ceil(np.max(self.EMDComb[0, :])))
        ax1.set_ylim(bottom=0, top=self.thN * self.thI);

        fig.savefig(figure_path, bbox_inches='tight')

        print('done.', flush=True)
        print(' ', flush=True)

    def plotSingleThresholdScatterPlot(self,
                                     dpi=300,
                                     pxx=2500,
                                     pxy=2500,
                                     padv=50,
                                     padh=50,
                                     padv2=0,
                                     padh2=0):
        """

        :param dpi: Dots per inch in figure
        :param pxx: Horizontal size of each panel in pixels
        :param pxy: Vertical size of each panel in pixels
        :param padv: Vertical padding in pixels at the edge of the figure in pixels
        :param padh: Horizontal padding in pixels at the edge of the figure in pixels
        :param padv2: Vertical padding in pixels between panels
        :param padh2: Horizontal padding in pixels between panels
        """

        font = self.font
        plt.rc('font', **font)

        print('Creating and saving scatterplot of overlap...', end="",
              flush=True)

        figure_path = self.output_path + '/' + str(self.CalObs) + '_' + self.NamObs + '/09_' + str(
            self.CalObs) + '_' + self.NamObs + '_Single_Threshold_ScatterPlot.png'

        nph = 1  # Number of horizontal panels
        npv = 1  # Number of vertical panels

        # Figure sizes in pixels
        fszv = (npv * pxy + 2 * padv + (npv - 1) * padv2)  # Vertical size of figure in inches
        fszh = (nph * pxx + 2 * padh + (nph - 1) * padh2)  # Horizontal size of figure in inches

        # Conversion to relative unites
        ppadv = padv / fszv  # Vertical padding in relative units
        ppadv2 = padv2 / fszv  # Vertical padding in relative units
        ppadh = padh / fszv  # Horizontal padding the edge of the figure in relative units
        ppadh2 = padh2 / fszv  # Horizontal padding between panels in relative units

        ## Start Figure
        fig = plt.figure(figsize=(fszh / dpi, fszv / dpi))

        # Calculating number of groups in reference data for given threshold
        grpsREFw = np.nansum(np.greater(self.REF_Dat.values[:, 3:self.REF_Dat.values.shape[1] - 3], self.wAv),
                             axis=1).astype(float)
        grpsREFw[np.isnan(self.REF_Dat['AREA1'])] = np.nan

        # Selecting the days of overlap with calibrated observer
        grpsREFw = grpsREFw[np.in1d(self.REF_Dat['ORDINAL'].values, self.ObsDat['ORDINAL'].values)]
        grpsObsw = self.ObsDat.loc[
            np.in1d(self.ObsDat['ORDINAL'].values, self.REF_Dat['ORDINAL'].values), 'GROUPS'].values

        # Removing NaNs
        grpsREFw = grpsREFw[np.isfinite(grpsObsw)]
        grpsObsw = grpsObsw[np.isfinite(grpsObsw)]

        grpsObsw = grpsObsw[np.isfinite(grpsREFw)]
        grpsREFw = grpsREFw[np.isfinite(grpsREFw)]

        maxN = np.max([np.nanmax(grpsREFw), np.nanmax(self.ObsDat['GROUPS'].values)])

        # Average group number
        ax1 = fig.add_axes([ppadh, ppadv, pxx / fszh, pxy / fszv])
        ax1.hist2d(grpsObsw, grpsREFw, bins=np.arange(0, np.ceil(maxN) + 1) - 0.5, cmap=plt.cm.magma_r, cmin=1)
        ax1.scatter(grpsObsw, grpsREFw, color='k', edgecolor='k', s=10, linewidths=3, zorder=11, alpha=0.01)
        ax1.plot(np.array([-0.5, maxN]), np.array([-0.5, maxN]), '--'
                 , color=self.Clr[4], linewidth=3)

        ax1.text(0.5, 0.95, '$R^2$ = ' + str(np.round(self.rSq, decimals=2)) + '  Mean Residual = ' + str(
            np.round(self.mRes, decimals=2)),
                 horizontalalignment='center', verticalalignment='center', transform=ax1.transAxes)

        # Axes properties
        ax1.set_xlabel('Observed GN for ' + self.NamObs)
        ax1.set_ylabel('Reference GN with threshold of ' + str(np.round(self.wAv, decimals=2)))


        fig.savefig(figure_path, bbox_inches='tight')

        print('done.', flush=True)
        print(' ', flush=True)


    def plotMultiThresholdScatterPlot(self,
                                     dpi=300,
                                     pxx=2500,
                                     pxy=2500,
                                     padv=50,
                                     padh=50,
                                     padv2=0,
                                     padh2=0):
        """

        :param dpi: Dots per inch in figure
        :param pxx: Horizontal size of each panel in pixels
        :param pxy: Vertical size of each panel in pixels
        :param padv: Vertical padding in pixels at the edge of the figure in pixels
        :param padh: Horizontal padding in pixels at the edge of the figure in pixels
        :param padv2: Vertical padding in pixels between panels
        :param padh2: Horizontal padding in pixels between panels
        """

        font = self.font
        plt.rc('font', **font)

        tcalRef = np.concatenate(self.calRef, axis=0)
        tcalObs = np.concatenate(self.calObs, axis=0)

        # Only if there is at leas one interval that is valid
        if tcalRef.shape[0] > 1:

            print('Creating and saving scatterplot of overlap with different thresholds...', end="",
                  flush=True)

            figure_path = self.output_path + '/' + str(self.CalObs) + '_' + self.NamObs + '/10_' + str(
                self.CalObs) + '_' + self.NamObs + '_Multi_Threshold_ScatterPlot.png'

            nph = 1  # Number of horizontal panels
            npv = 1  # Number of vertical panels

            nph = 1  # Number of horizontal panels
            npv = 1  # Number of vertical panels

            # Figure sizes in pixels
            fszv = (npv * pxy + 2 * padv + (npv - 1) * padv2)  # Vertical size of figure in inches
            fszh = (nph * pxx + 2 * padh + (nph - 1) * padh2)  # Horizontal size of figure in inches

            # Conversion to relative unites
            ppadv = padv / fszv  # Vertical padding in relative units
            ppadv2 = padv2 / fszv  # Vertical padding in relative units
            ppadh = padh / fszv  # Horizontal padding the edge of the figure in relative units
            ppadh2 = padh2 / fszv  # Horizontal padding between panels in relative units

            ## Start Figure
            fig = plt.figure(figsize=(fszh / dpi, fszv / dpi))

            # Calculate R^2 and residual using only valid periods
            calRefN = np.array([0])
            calObsN = np.array([0])

            for n in range(0, self.cenPoints['OBS'].shape[0]):

                # Plot only if the period is valid and has overlap
                if self.vldIntr[n] and np.sum(np.logical_and(self.REF_Dat['FRACYEAR'] >= self.endPoints['OBS'][n, 0],
                                                             self.REF_Dat['FRACYEAR'] < self.endPoints['OBS'][
                                                                 n + 1, 0])) > 0:
                    # Calculating number of groups in reference data for given threshold
                    grpsREFw = np.nansum(np.greater(self.REF_Dat.values[:, 3:self.REF_Dat.values.shape[1] - 3], self.wAv),
                                         axis=1).astype(float)
                    grpsREFw[np.isnan(self.REF_Dat['AREA1'])] = np.nan

                    # Selecting observer's interval
                    TObsDat = self.ObsDat.loc[
                        np.logical_and(self.ObsDat['FRACYEAR'] >= self.endPoints['OBS'][n, 0],
                                       self.ObsDat['FRACYEAR'] < self.endPoints['OBS'][n + 1, 0])
                        , 'GROUPS'].values.copy()
                    TObsOrd = self.ObsDat.loc[
                        np.logical_and(self.ObsDat['FRACYEAR'] >= self.endPoints['OBS'][n, 0],
                                       self.ObsDat['FRACYEAR'] < self.endPoints['OBS'][n + 1, 0])
                        , 'ORDINAL'].values.copy()

                    # Selecting the days of overlap with calibrated observer
                    grpsREFw = grpsREFw[np.in1d(self.REF_Dat['ORDINAL'].values, TObsOrd)]
                    grpsObsw = TObsDat[np.in1d(TObsOrd, self.REF_Dat['ORDINAL'].values)]

                    # Removing NaNs
                    grpsREFw = grpsREFw[np.isfinite(grpsObsw)]
                    grpsObsw = grpsObsw[np.isfinite(grpsObsw)]

                    grpsObsw = grpsObsw[np.isfinite(grpsREFw)]
                    grpsREFw = grpsREFw[np.isfinite(grpsREFw)]

                    # Appending to calibrated arrays?
                    calRefN = np.append(calRefN, grpsREFw)
                    calObsN = np.append(calObsN, grpsObsw)

            maxN = np.max([np.max(calRefN), np.max(calObsN), np.max(tcalRef), np.max(tcalObs)])

            ax1 = fig.add_axes([ppadh, ppadv, pxx / fszh, pxy / fszv], label='b1')

            # Average group number
            ax1.hist2d(calObsN, calRefN, bins=np.arange(0, np.ceil(maxN) + 1) - 0.5, cmap=plt.cm.magma_r, cmin=1)
            ax1.scatter(calObsN, calRefN, color='k', edgecolor='k', s=10, linewidths=3, zorder=11, alpha=0.01)
            ax1.plot(np.array([-0.5, maxN]), np.array([-0.5, maxN]), '--'
                     , color=self.Clr[4], linewidth=3)

            ax1.text(0.5, 0.95, '$R^2$ = ' + str(np.round(self.rSqOO, decimals=2)) + '  Mean Residual = ' + str(
                np.round(self.mResOO, decimals=2)), horizontalalignment='center', verticalalignment='center',
                     transform=ax1.transAxes)

            # Axes properties
            ax1.set_xlabel('Observed GN for ' + self.NamObs)
            ax1.set_ylabel('Reference GN with threshold of ' + str(np.round(self.wAv, decimals=2)))

            # Average group number
            ax1 = fig.add_axes([ppadh + (pxx / fszh + ppadh2), ppadv, pxx / fszh, pxy / fszv], label='b2')
            ax1.hist2d(tcalObs, tcalRef, bins=np.arange(0, np.ceil(maxN) + 1) - 0.5, cmap=plt.cm.magma_r, cmin=1)
            ax1.scatter(tcalObs, tcalRef, color='k', edgecolor='k', s=10, linewidths=3, zorder=11, alpha=0.01)
            ax1.plot(np.array([-0.5, maxN]), np.array([-0.5, maxN]), '--'
                     , color=self.Clr[4], linewidth=3)

            ax1.text(0.5, 0.95,
                     '$R^2$ = ' + str(np.round(self.rSqDT, decimals=2)) + '  Mean Residual = ' + str(np.round(self.mResDT, decimals=2)),
                     horizontalalignment='center', verticalalignment='center', transform=ax1.transAxes)

            # Axes properties
            ax1.set_xlabel('Observed GN for ' + self.NamObs)
            ax1.set_ylabel('Reference GN with different thresholds')

            fig.savefig(figure_path, bbox_inches='tight')

            print('done.', flush=True)
            print(' ', flush=True)

