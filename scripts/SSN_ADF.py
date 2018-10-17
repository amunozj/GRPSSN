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
from SSN_Config import SSN_ADF_Config as config

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
                 silso_path_daily='input_data/SN_d_tot_V2.0.csv',
                 obs_data_path='input_data/GNObservations_JV_V1.22.csv',
                 obs_observer_path='input_data/GNObservers_JV_V1.22.csv',
                 output_path='output',
                 font=None,
                 MoLngt=15,
                 dt=10,
                 phTol=2,
                 thN=100,
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
        REF_Grp['AVGROUPS'] = conv.convolve(REF_Grp['GROUPS'].values, Gss_1D_ker)

        print('done.', flush=True)

        # ----------------------------------------------------------------------------------------------------
        print('Reading SILSO Data...', end="", flush=True)

        SILSO_Sn = pd.read_csv(silso_path, quotechar='"', encoding='utf-8', header=0)
        SILSO_Sn_d = pd.read_csv(silso_path_daily, quotechar='"', encoding='utf-8', header=0)
        
        # Including daily value and interpolating
        SILSO_Sn_d['MONTHSN']=SILSO_Sn_d['DAILYSN']
        SILSO_Sn_d.loc[SILSO_Sn_d['DAILYSN']<0,'MONTHSN'] = np.interp(SILSO_Sn_d.loc[SILSO_Sn_d['DAILYSN']<0,'FRACYEAR'],
                                                                      SILSO_Sn_d.loc[SILSO_Sn_d['DAILYSN']>=0,'FRACYEAR'],
                                                                      SILSO_Sn_d.loc[SILSO_Sn_d['DAILYSN']>=0,'DAILYSN'])
        SILSO_Sn_d['DAILYSN'] = SILSO_Sn_d['DAILYSN'].astype(float)
        
        # Smoothing for plotting
        Gss_1D_ker = conv.Gaussian1DKernel(365)
        SILSO_Sn_d['AVGSNd']  = conv.convolve(SILSO_Sn_d['DAILYSN'].values,Gss_1D_ker)

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
        cond2=pd.merge(SILSO_Sn_d, REF_Grp, on=['YEAR', 'MONTH', 'DAY'], how='inner')
        REF_Grp=REF_Grp.join(cond2['AVGSNd'])

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
        # --------------------------------------------------------------------------------------------------
        print('Creating plot SN vs ADF...', end="", flush=True)
        
        ssn_data.MoLngt = MoLngt
#         MoLngt = 15
        thNP = 21  # Number of thresholds to plot
        thSP = 5  # Threshold increment

        # creating matrix to define thresholds
        TREFDat = REF_Grp['GROUPS'].values.copy()
        TREFSNd = REF_Grp['AVGSNd'].values.copy()

        GDREF = np.zeros((thNP,np.int(TREFDat.shape[0]/MoLngt)))
        ODREF = np.zeros((thNP,np.int(TREFDat.shape[0]/MoLngt)))
        SNdREF = np.zeros((thNP,np.int(TREFDat.shape[0]/MoLngt)))

        for TIdx in range(0,thNP):
                    grpsREFw = np.nansum( np.greater(REF_Dat.values[:,3:REF_Dat.values.shape[1]-3],TIdx*thSP) ,axis = 1).astype(float)
                    grpsREFw[np.isnan(REF_Dat['AREA1'])] = np.nan

                    TgrpsREF = grpsREFw[0:np.int(grpsREFw.shape[0]/MoLngt)*MoLngt].copy()
                    TgrpsREF = TgrpsREF.reshape((-1,MoLngt))            
                    TSNdREF = TREFSNd[0:np.int(TREFSNd.shape[0]/MoLngt)*MoLngt].copy()
                    TSNdREF = TSNdREF.reshape((-1,MoLngt))            
                    # Number of days with groups
                    GDREF[TIdx,:] = np.sum(np.greater(TgrpsREF,0),axis=1)
                    # Number of days with observations
                    ODREF[TIdx,:]= np.sum(np.isfinite(TgrpsREF),axis=1)            
                    # Number of quiet days
                    QDREF = ODREF-GDREF
                    # ACTIVE DAY FRACTION
                    ADFREF = GDREF/ODREF
                    # Monthly sunspot number
                    SNdREF[TIdx,:] = np.mean(TSNdREF,axis=1)     
                    

        bprange = np.arange(10,175,10)
        pprange = np.arange(5,175,2)

        LowALlim = np.zeros(thNP)
        HighALlim = np.zeros(thNP)

        for n in range(0,thNP):

                pltmsk = np.logical_and(ODREF[n,:]==MoLngt,ADFREF[n,:]<1)

                bpdat = []    
                for AL in bprange:
                    bpdat.append(ADFREF[n,:][np.logical_and(pltmsk, SNdREF[n,:]<=AL)])

                ALP = pprange*np.nan
                for ALi in np.arange(0,pprange.shape[0]):
                    if (np.sum(np.logical_and(pltmsk, SNdREF[n,:]<=pprange[ALi]))>0):ALP[ALi] = np.percentile(ADFREF[n,:][np.logical_and(pltmsk, SNdREF[n,:]<=pprange[ALi])], config.PCTLO)

                intrsc = np.where(np.abs(ALP-0.25)==np.nanmin(np.abs(ALP-0.25)))[0]
                cut = np.mean(pprange[intrsc])        
                if np.sum(ALP<0.25)==0:
                    cut = np.nan

                LowALlim[n] = cut

                #ax2

                bpdat = []
                for AL in bprange:        
                    bpdat.append(ADFREF[n,:][np.logical_and(pltmsk, SNdREF[n,:]>=AL)])

                ALP = pprange*np.nan
                for ALi in np.arange(0,pprange.shape[0]):
                    if (np.sum(np.logical_and(pltmsk, SNdREF[n,:]>=pprange[ALi]))>0):ALP[ALi] = np.percentile(ADFREF[n,:][np.logical_and(pltmsk, SNdREF[n,:]>=pprange[ALi])], 100-config.PCTHI)

                intrsc = np.where(np.abs(ALP-0.75)==np.nanmin(np.abs(ALP-0.75)))[0]
                cut = np.mean(pprange[intrsc])
                if np.sum(ALP<0.75)==0:
                    cut = np.nan

                HighALlim[n] = cut
                
        
        # fit for low solar activity
        xlow = np.arange(0,thNP)*thSP
        xlow = xlow[np.isfinite(LowALlim)]
        ylow = LowALlim[np.isfinite(LowALlim)]
        fitlow = np.polyfit(xlow,ylow,deg=1)
        a1low = fitlow[0]
        a0low = fitlow[1]

        # fit for high solar activity
        xhigh = np.arange(0,thNP)*thSP
        xhigh = xhigh[np.isfinite(HighALlim)]
        yhigh = HighALlim[np.isfinite(HighALlim)]
        fithigh = np.polyfit(xhigh,yhigh,deg=1)
        a1high = fithigh[0]
        a0high = fithigh[1]
                    

        # Storing variables in object-----------------------------------------------------------------------------------

        self.ssn_data.output_path = output_path  # Location of all output files
        
        self.ssn_data.MoLngt = MoLngt  # Duration of the interval ("month") used to calculate the ADF

        self.ssn_data.font = font  # Font to be used while plotting
        self.ssn_data.ssn_datadt = dt  # Temporal Stride in days
        self.ssn_data.phTol = phTol  # Cycle phase tolerance in years
        self.ssn_data.thN = thN  # Number of thresholds including 0
        self.ssn_data.thI = thI  # Threshold increments
        
        self.ssn_data.thNP = thNP  # Number of thresholds including 0
        self.ssn_data.thSP = thSP  # Threshold increments
        self.ssn_data.pltmsk = pltmsk  # mask for plot
        self.ssn_data.bpdat = bpdat  
        self.ssn_data.ALP = ALP
        self.ssn_data.cut = cut
        

        self.ssn_data.REF_Dat = REF_Dat  # Reference data with individual group areas each day
        self.ssn_data.REF_Grp = REF_Grp  # Reference data with individual numbers of sunspot for each day
        self.ssn_data.SILSO_Sn_d = SILSO_Sn_d  # SILSO data for each day
        
        self.ssn_data.SNdREF = SNdREF  # Sunspot number from reference to plot
        self.ssn_data.ADFREF = ADFREF  # ADF from reference to plot
        

        self.ssn_data.risMask = risMask  # Mask indicating where to place the search window during raising phases
        self.ssn_data.decMask = decMask  # Mask indicating where to place the search window during declining phases

        self.ssn_data.endPoints = {
            'SILSO': endPointsS}  # Variable that stores the boundaries of each rising and decaying phase
        self.ssn_data.cenPoints = {
            'SILSO': cenPointsS}  # Variable that stores the centers of each rising and decaying phase
        
        
        
        self.ssn_data.LowALlim = LowALlim # Data to obtain the coefficients for the fits of the low solar activity
        self.ssn_data.HighALlim = HighALlim # Data to obtain the coefficients for the fits of the high solar activity
        
        self.ssn_data.xlow = xlow # Data of the x-axis for the fit to obtain the threshold for low solar activity
        self.ssn_data.ylow = ylow # Data of the y-axis for the fit to obtain the threshold for low solar activity  
        self.ssn_data.xhigh = xhigh 
        self.ssn_data.yhigh = yhigh    
        self.ssn_data.fitlow = fitlow 
        self.ssn_data.fithigh = fithigh        
        
        self.ssn_data.a1high = a1high # Coefficient #1 of the fit for high solar activity
        self.ssn_data.a0high = a0high # Coefficient #0 of the fit for high solar activity
        self.ssn_data.a1low = a1low # Coefficient #1 of the fit for low solar activity
        self.ssn_data.a0low = a0low # Coefficient #0 of the fit for low solar activity   


        # --------------------------------------------------------------------------------------------------------------

        if plot:
            SSN_ADF_Plotter.plotSearchWindows(self.ssn_data, SILSO_Sn, SILSO_Sn_d, SIL_max, SIL_min, REF_min, REF_max)

        print('Done initializing data.', flush=True)
        print(' ', flush=True)
        
        # Plot histogram SN vs ADF and scatter SN vs AL
        if plot:
            SSN_ADF_Plotter.plotHistSnADF(self.ssn_data)

        print('Done initializing data.', flush=True)
        print(' ', flush=True)
        
        if plot:
            SSN_ADF_Plotter.plotFitAl(self.ssn_data)

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

        # Removing repeated days
        u, indices = np.unique(ObsDat['ORDINAL'], return_index=True)
        ObsDat = ObsDat.iloc[indices,:].reset_index(drop=True)
        
        # Attaching daily SN to observer data
        cond1=pd.merge(ssn_data.SILSO_Sn_d, ObsDat, on=['YEAR', 'MONTH', 'DAY'], how='inner')
        ObsDat=ObsDat.join(cond1['AVGSNd'])

        print('Calculating variables for plotting observer...', flush=True)

        # Selecting the maximum integer amount of "months" out of the original data
        yrOb = ObsDat['FRACYEAR'].values
        yrOb = yrOb[0:np.int(yrOb.shape[0] / MoLngt) * MoLngt]
        
        ordOb = ObsDat['ORDINAL'].values
        ordOb = ordOb[0:np.int(ordOb.shape[0]/MoLngt)*MoLngt]

        grpsOb = ObsDat['GROUPS'].values
        grpsOb = grpsOb[0:np.int(grpsOb.shape[0] / MoLngt) * MoLngt]
        
        SNdOb = ObsDat['AVGSNd'].values
        SNdOb = SNdOb[0:np.int(SNdOb.shape[0]/MoLngt)*MoLngt]

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
        ordOb  = ordOb.reshape((-1,MoLngt))
        SNdOb  = SNdOb.reshape((-1,MoLngt))

        # Interval edges for plotting
        fyr1Ob = np.min(yrOb, axis=1)
        fyr2Ob = np.max(yrOb, axis=1)

        obsPlt = {'X': (fyr1Ob + fyr2Ob) / 2}

        # Average number of groups
        Gss_1D_ker = conv.Gaussian1DKernel(2)
        obsPlt['Y'] = conv.convolve(np.nanmean(grpsOb, axis=1), Gss_1D_ker)
        SdGrpOb = np.nanstd(grpsOb,axis=1)

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
        ssn_data.QDays = n_qui  # Total number of Quiet days
        ssn_data.ADays = n_act  # Total number of Active days
        ssn_data.NADays = n_na  # Total number of missing days in data
        ssn_data.QAFrac = round(n_qui / n_act, 3)  # Fraction of quiet to active days

        ssn_data.ObsPerMonth = round((n_act + n_qui) / yrOb.shape[0], 3)  # Average number of days observed per month

        ssn_data.RiseCount = len([x for x in cenPoints if x[1] == 1.0])  # Number of intervals in rising phases
        ssn_data.DecCount = len([x for x in cenPoints if x[1] == -1.0])  # Number of intervals in declining phases

        ssn_data.InvCount = np.sum(np.logical_not(vldIntr))  # Number of invalid intervals in observer

        ssn_data.InvMonths = np.sum(np.logical_not(vMonths)) # Number of invalid months in observer
        moStrk = [sum(1 for _ in g) for k, g in groupby(vMonths) if not k]
        if moStrk:
            ssn_data.InvMoStreak = max(moStrk)  # Highest number of invalid months in a row (biggest gap)
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

    def _Calculate_R2M_MRes_MRRes(self,
                                 calObsT,
                                 calRefT,
                                 centers,
                                 edges):

        """
        Function that calculates the R^2 and mean residuals using the medians of binned data.

        :param calObsT: Number of groups per day for the calibrated observer
        :param MaxIter: Number of groups per day for the reference, these are meant to match
        one by one the same dates of the calibrated observer and to be already thresholded
        :param centers: Centers of the bins used in the calibration
        :param edges: Edges of the bins used in the calibration
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

        # Calculating quantities for assessment
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

        return {'rSq': rSq,
                'mRes': mRes,
                'mRRes': mRRes}

    def ADFscanningWindowEMD(self,
                             ssn_data,
                             Dis_Pow = 2):

        """
        Function that preps the search windows and calculates the EMD for each separate rising and decaying interval
        comparing the observer and the reference
        VARIABLES APPENDED TO THE OBJECT ARE SPECIFIED AT THE END

        :param ssn_data: ssn_data class object storing SSN metadata
        :param Dis_Pow: Power index used to define the distance matrix for EMD calculation
        :return:  (False) True if there are (no) valid days of overlap between observer and reference
        """

        print('Calculating number of active and observed days using scanning windows...', flush=True)   
               

        thN = 21  # Number of thresholds to plot
        thS = 5  # Threshold increment

        # creating matrix to define thresholds
        TREFDat = ssn_data.REF_Grp['GROUPS'].values.copy()
        TREFSNd = ssn_data.REF_Grp['AVGSNd'].values.copy()

        GDREF = np.zeros((thN,np.int(TREFDat.shape[0]/ssn_data.MoLngt)))
        ODREF = np.zeros((thN,np.int(TREFDat.shape[0]/ssn_data.MoLngt)))
        SNdREF = np.zeros((thN,np.int(TREFDat.shape[0]/ssn_data.MoLngt)))

        for TIdx in range(0,thN):
                    grpsREFw = np.nansum( np.greater(ssn_data.REF_Dat.values[:,3:ssn_data.REF_Dat.values.shape[1]-3],TIdx*thS) ,axis = 1).astype(float)
                    grpsREFw[np.isnan(ssn_data.REF_Dat['AREA1'])] = np.nan

                    TgrpsREF = grpsREFw[0:np.int(grpsREFw.shape[0]/ssn_data.MoLngt)*ssn_data.MoLngt].copy()
                    TgrpsREF = TgrpsREF.reshape((-1,ssn_data.MoLngt))            
                    TSNdREF = TREFSNd[0:np.int(TREFSNd.shape[0]/ssn_data.MoLngt)*ssn_data.MoLngt].copy()
                    TSNdREF = TSNdREF.reshape((-1,ssn_data.MoLngt))            
                    # Number of days with groups
                    GDREF[TIdx,:] = np.sum(np.greater(TgrpsREF,0),axis=1)
                    # Number of days with observations
                    ODREF[TIdx,:]= np.sum(np.isfinite(TgrpsREF),axis=1)            
                    # Number of quiet days
                    QDREF = ODREF-GDREF
                    # ACTIVE DAY FRACTION
                    ADFREF = GDREF/ODREF
                    # Monthly sunspot number
                    SNdREF[TIdx,:]=np.mean(TSNdREF,axis=1) 


        bprange = np.arange(10,175,10)
        pprange = np.arange(5,175,2)

        LowALlim = np.zeros(thN)
        HighALlim = np.zeros(thN)

        for n in range(0,thN):

                    pltmsk = np.logical_and(ODREF[n,:]==ssn_data.MoLngt,ADFREF[n,:]<1)

                    bpdat = []    
                    for AL in bprange:
                        bpdat.append(ADFREF[n,:][np.logical_and(pltmsk, SNdREF[n,:]<=AL)])

                    ALP = pprange*np.nan
                    for ALi in np.arange(0,pprange.shape[0]):
                        if (np.sum(np.logical_and(pltmsk, SNdREF[n,:]<=pprange[ALi]))>0):ALP[ALi] = np.percentile(ADFREF[n,:][np.logical_and(pltmsk, SNdREF[n,:]<=pprange[ALi])], config.PCTLO)

                    intrsc = np.where(np.abs(ALP-0.25)==np.nanmin(np.abs(ALP-0.25)))[0]
                    cut = np.mean(pprange[intrsc])        
                    if np.sum(ALP<0.25)==0:
                        cut = np.nan

                    LowALlim[n] = cut

                    bpdat = []
                    for AL in bprange:        
                        bpdat.append(ADFREF[n,:][np.logical_and(pltmsk, SNdREF[n,:]>=AL)])

                    ALP = pprange*np.nan
                    for ALi in np.arange(0,pprange.shape[0]):
                        if (np.sum(np.logical_and(pltmsk, SNdREF[n,:]>=pprange[ALi]))>0):ALP[ALi] = np.percentile(ADFREF[n,:][np.logical_and(pltmsk, SNdREF[n,:]>=pprange[ALi])], 100-config.PCTHI)

                    intrsc = np.where(np.abs(ALP-0.75)==np.nanmin(np.abs(ALP-0.75)))[0]
                    cut = np.mean(pprange[intrsc])
                    if np.sum(ALP<0.75)==0:
                        cut = np.nan

                    HighALlim[n] = cut

        
        # fit for low solar activity
        xlow = np.arange(0,thN)*thS
        xlow = xlow[np.isfinite(LowALlim)]
        ylow = LowALlim[np.isfinite(LowALlim)]
        fitlow = np.polyfit(xlow,ylow,deg=1)
        a1low = fitlow[0]
        a0low = fitlow[1]

        # fit for high solar activity
        xhigh = np.arange(0,thN)*thS
        xhigh = xhigh[np.isfinite(HighALlim)]
        yhigh = HighALlim[np.isfinite(HighALlim)]
        fithigh = np.polyfit(xhigh,yhigh,deg=1)
        a1high = fithigh[0]
        a0high = fithigh[1]


        # Number of thresholds including 0
        thN = 100
        
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
        
        # Monthly (from daily) sunspot number
        SNdObsI = []
        SNdREFI = []
        
#         #creating storing dictionaries for ADF
#         ADF_Obs_fracI = []
#         ADF_REF_fracI = []

        # Number of days rising or declining
        rise_count = 0
        dec_count = 0

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
                TObsSNd = ssn_data.ObsDat.loc[
                    np.logical_and(ssn_data.ObsDat['FRACYEAR']>=ssn_data.endPoints['OBS'][siInx, 0],
                                   ssn_data.ObsDat['FRACYEAR']<ssn_data.endPoints['OBS'][siInx + 1, 0])
                             ,'AVGSNd'].values.copy()                

                # Selecting the maximum integer amount of "months" out of the original data
                TgrpsOb = TObsDat[0:np.int(TObsDat.shape[0] / ssn_data.MoLngt) * ssn_data.MoLngt].copy()
                TgrpsOb = TgrpsOb.reshape((-1, ssn_data.MoLngt))
                TgrpsOb = np.logical_not(np.isnan(TgrpsOb))
                for m in TgrpsOb:
                    if np.sum(m) / ssn_data.MoLngt >= ssn_data.minObD:
                        if ssn_data.cenPoints['OBS'][siInx, 1] == 1.0:
                            rise_count += 1
                        elif ssn_data.cenPoints['OBS'][siInx, 1] == -1.0:
                            dec_count += 1

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
                
                # mask for monthly (from daily) sunspot number
                SNdObs = np.zeros((ssn_data.thN,cadMaskI.shape[0],np.int(TObsSNd.shape[0]/ssn_data.MoLngt))) 
                SNdREF = np.zeros((ssn_data.thN,cadMaskI.shape[0],np.int(TObsSNd.shape[0]/ssn_data.MoLngt)))
                    
                # Going through different thresholds
                for TIdx in range(0, ssn_data.thN):

                    # Calculating number of groups in reference data for given threshold
                    grpsREFw = np.nansum(
                        np.greater(ssn_data.REF_Dat.values[:, 3:ssn_data.REF_Dat.values.shape[1] - 3],
                                   TIdx * ssn_data.thI),
                        axis=1).astype(float)
                    grpsREFw[np.isnan(ssn_data.REF_Dat['AREA1'])] = np.nan
                    
                    # Final fit to define threshold
                    highth = a1high*TIdx * ssn_data.thI + a0high
                    if TIdx * ssn_data.thI >= np.min(xlow): 
                        lowth = a1low*TIdx * ssn_data.thI + a0low
                    else:
                        lowth = 0            

                    # Going through different shifts
                    for SIdx in range(0, cadMaskI.shape[0]):

                        # Selecting the maximum integer amount of "months" out of the original data
                        TgrpsOb = TObsDat[0:np.int(TObsDat.shape[0] / ssn_data.MoLngt) * ssn_data.MoLngt].copy()
                        
                        TObsSNd = TObsSNd[0:np.int(TObsDat.shape[0]/ssn_data.MoLngt)*ssn_data.MoLngt].copy()

                        # Calculating bracketing indices
                        Idx1 = cadMaskI[SIdx] - obsMinInx
                        Idx2 = Idx1 + TgrpsOb.shape[0]

                        # Selecting reference window of matching size to observer sub-interval;
                        TgrpsREF = grpsREFw[Idx1:Idx2].copy()
                        
                        TSNdREF = ssn_data.REF_Grp['AVGSNd'][Idx1:Idx2].values.copy()

                        # Making sure selections have the same length
                        if TgrpsREF.shape[0] == TgrpsOb.shape[0]:
                            # Reshaping into "months"
                            TgrpsOb = TgrpsOb.reshape((-1, ssn_data.MoLngt))
                            TgrpsREF = TgrpsREF.reshape((-1, ssn_data.MoLngt))
                            # Reshaping SN into "months"
                            TObsSNd = TObsSNd.reshape((-1,ssn_data.MoLngt))
                            TSNdREF = TSNdREF.reshape((-1,ssn_data.MoLngt))                            

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
                            
                            # monthly sunspot number
                            SNdObs[TIdx,SIdx,:] = np.mean(TObsSNd,axis=1)
                            SNdREF[TIdx,SIdx,:] = np.mean(TSNdREF,axis=1)


            # If period is not valid append empty variavbles
            else:
                print('[{}/{}] INVALID Interval'.format(siInx + 1, num_intervals))
                GDObs = []
                GDREF = []
                ODObs = []
                ODREF = []
                QDObs = []
                QDREF = []                
                SNdObs = []
                SNdREF = []
              

            print(' ')

            # Appending calculated days to list of sub-intervals
            GDObsI.append(GDObs)
            GDREFI.append(GDREF)
            ODObsI.append(ODObs)
            ODREFI.append(ODREF)
            QDObsI.append(QDObs)
            QDREFI.append(QDREF)                       
            SNdObsI.append(SNdObs)    
            SNdREFI.append(SNdREF)
        

        print('done.', flush=True)
        print(' ', flush=True)

        print('Calculating the Earths Mover Distance using a sliding window...', flush=True)

        # Creating Storing dictionaries for distance matrices
        EMDD = []
        EMDiD = []
        EMDtD = []
        EMDthD = []
        EMDthiD = []

        # Calculation of distance matrix to be used in the Earth Movers Metric
        x = np.arange(0, ssn_data.MoLngt + 1)
        y = np.arange(0, ssn_data.MoLngt + 1)
        xx, yy = np.meshgrid(x, y)
        Dis = np.absolute(np.power(xx - yy, Dis_Pow))

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
                EMDi = np.zeros((GDREFI[siInx].shape[0], GDREFI[siInx].shape[1]))
                EMDt = np.zeros((GDREFI[siInx].shape[0], GDREFI[siInx].shape[1]))
                EMDth = np.zeros((GDREFI[siInx].shape[0], GDREFI[siInx].shape[1]))
                EMDthi = np.zeros((GDREFI[siInx].shape[0], GDREFI[siInx].shape[1]))

                # Going through different thresholds
                for TIdx in range(0, ssn_data.thN):

                    # Going through different shifts
                    for SIdx in range(0, cadMaskI.shape[0]):
                    

                        if np.any(ODObsI[siInx][TIdx, SIdx, :] != 0) and np.any(ODREFI[siInx][TIdx, SIdx, :] != 0):
                            
                            # Calculating Earth Mover's Distance 
                            
                            # Numerator and denominator for given observer
                            numADObs = GDObsI[siInx][TIdx,SIdx,ODObsI[siInx][TIdx,SIdx,:]/ssn_data.MoLngt>=ssn_data.minObD]
                            numQDObs = ssn_data.MoLngt - QDObsI[siInx][TIdx,SIdx,ODObsI[siInx][TIdx,SIdx,:]/ssn_data.MoLngt>=ssn_data.minObD] 
                            denFMObs = GDObsI[siInx][TIdx,SIdx,ODObsI[siInx][TIdx,SIdx,:]/ssn_data.MoLngt>=ssn_data.minObD]*0 + ssn_data.MoLngt
                            denODObs = ODObsI[siInx][TIdx,SIdx,ODObsI[siInx][TIdx,SIdx,:]/ssn_data.MoLngt>=ssn_data.minObD]
                            # Numerator and denominator for reference
                            numADREF = GDREFI[siInx][TIdx,SIdx,ODREFI[siInx][TIdx,SIdx,:]/ssn_data.MoLngt>=ssn_data.minObD]
                            numQDREF = ssn_data.MoLngt - QDREFI[siInx][TIdx,SIdx,ODREFI[siInx][TIdx,SIdx,:]/ssn_data.MoLngt>=ssn_data.minObD]
                            denFMREF = GDREFI[siInx][TIdx,SIdx,ODREFI[siInx][TIdx,SIdx,:]/ssn_data.MoLngt>=ssn_data.minObD]*0 + ssn_data.MoLngt
                            denODREF = ODREFI[siInx][TIdx,SIdx,ODREFI[siInx][TIdx,SIdx,:]/ssn_data.MoLngt>=ssn_data.minObD] 
                            
                            if config.NUM_TYPE == "ADF": 
                                numObs = numADObs
                                numREF = numADREF
                            else: 
                                numObs = numQDObs
                                numREF = numQDREF
                                
                            if config.DEN_TYPE == "OBS":
                                denObs = denODObs
                                denREF = denODREF
                            else:
                                denObs = denFMObs
                                denREF = denFMREF

                            
                            if config.DEN_TYPE == "DTh":
                                #defining solar activity level                            
                                MMObs=np.logical_and((SNdObsI[siInx][TIdx,SIdx,ODObsI[siInx][TIdx,SIdx,:]/ssn_data.MoLngt>=ssn_data.minObD]>lowth), (SNdObsI[siInx][TIdx,SIdx,ODObsI[siInx][TIdx,SIdx,:]/ssn_data.MoLngt>=ssn_data.minObD]<highth))
                                MMREF=np.logical_and((SNdREFI[siInx][TIdx,SIdx,ODREFI[siInx][TIdx,SIdx,:]/ssn_data.MoLngt>=ssn_data.minObD]>lowth), (SNdREFI[siInx][TIdx,SIdx,ODREFI[siInx][TIdx,SIdx,:]/ssn_data.MoLngt>=ssn_data.minObD]<highth))

                                HMObs=(SNdObsI[siInx][TIdx,SIdx,ODObsI[siInx][TIdx,SIdx,:]/ssn_data.MoLngt>=ssn_data.minObD]>=highth)
                                HMREF=(SNdREFI[siInx][TIdx,SIdx,ODREFI[siInx][TIdx,SIdx,:]/ssn_data.MoLngt>=ssn_data.minObD]>=highth)  

                                # Default numerators and denominators
                                numObs = numADObs
                                numREF = numADREF
                                denObs = denFMObs
                                denREF = denFMREF


                                numObs[HMObs] = numQDObs[HMObs]
                                numREF[HMREF] = numQDREF[HMREF]

                                denObs[MMObs] = denODObs[MMObs]
                                denREF[MMREF] = denODREF[MMREF]

                            # ADF calculations
                            ADF_Obs_fracI = np.divide(numObs, denObs)
                            ADF_REF_fracI = np.divide(numREF, denREF)
                            
                            

                            # Main ADF calculations
                            ADFObs, bins = np.histogram(ADF_Obs_fracI,bins=(np.arange(0,ssn_data.MoLngt + 2) - 0.5) / ssn_data.MoLngt,density=True)

                            ADFREF, bins = np.histogram(ADF_REF_fracI,bins=(np.arange(0,ssn_data.MoLngt + 2) - 0.5) / ssn_data.MoLngt,density=True)
                            
                            EMD[TIdx, SIdx] = emd(ADFREF.astype(np.float64), ADFObs.astype(np.float64), Dis.astype(np.float64))
                            EMDi[TIdx, SIdx] = SIdx

                        # Storing coordinates of EMD distances
                        EMDt[TIdx, SIdx] = ssn_data.REF_Grp['FRACYEAR'].values[cadMaskI[SIdx]]
                        EMDth[TIdx, SIdx] = TIdx * ssn_data.thI
                        EMDthi[TIdx, SIdx] = TIdx

            # If period is not valid append empty variables
            else:
                print('[{}/{}] INVALID Interval'.format(siInx + 1, num_intervals))
                EMD = []
                EMDi = []
                EMDt = []
                EMDth = []
                EMDthi = []

            print(' ')

            EMDD.append(EMD)
            EMDiD.append(EMDi)
            EMDtD.append(EMDt)
            EMDthD.append(EMDth)
            EMDthiD.append(EMDthi)

            
        print('done.', flush=True)
        print(' ', flush=True)

        print('Identifying the best matches for each valid period and looking for ref-obs overlap...', end="",
              flush=True)

        # Creating Storing dictionaries to store best thresholds
        bestTh = []
        calRef = []
        calObs = []

        # Switch indicating that there is overlap between reference and observer
        obsRefOvrlp = False

        # Variables to store the mean threshold and its standard deviation
        wAvI = ssn_data.vldIntr.copy() * 0.0
        wSDI = wAvI.copy()

        # Calculating maximum for plotting, medians, and standard deviations
        maxNPlt = 0

        for siInx in range(0, ssn_data.cenPoints['OBS'].shape[0]):

            # Analyze period only if valid
            if ssn_data.vldIntr[siInx]:

                # Creating matrix for sorting and find the best combinations of threshold and shift
                OpMat = np.concatenate(
                    (EMDtD[siInx].reshape((-1, 1)), EMDthD[siInx].reshape((-1, 1)),
                     EMDD[siInx].reshape((-1, 1)), EMDiD[siInx].reshape((-1,1)),
                     EMDthiD[siInx].reshape((-1,1))), axis=1)

                # Sort according to EMD to find the best matches
                I = np.argsort(OpMat[:, 2], axis=0)
                OpMat = np.squeeze(OpMat[I, :])

                # Adding best points
                bestTh.append(OpMat[0:config.NBEST, :])

                if config.NBEST == 1:
                    alph = bestTh[siInx][:, 2] * 0 + 1
                else:
                    # Constructing weights
                    alph = 1 - (bestTh[siInx][:, 2] - np.min(bestTh[siInx][:, 2])) / (
                        np.max(bestTh[siInx][:, 2]) - np.min(bestTh[siInx][:, 2]))

                if np.isnan(np.sum(alph)):
                    alph = bestTh[siInx][:, 2] * 0 + 1

                # Weighted average
                wAvI[siInx] = np.sum(np.multiply(alph, bestTh[siInx][:, 1])) / np.sum(alph)

                # Weighted Standard Deviation
                if config.NBEST == 1:
                    wSDI[siInx] = np.nan
                else:
                    wSDI[siInx] = np.sqrt(np.sum(np.multiply(alph, np.power(bestTh[siInx][:, 1] - wAvI[siInx], 2))) / np.sum(alph))

                if np.sum(np.logical_and(ssn_data.REF_Dat['FRACYEAR'] > ssn_data.endPoints['OBS'][siInx, 0],
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

                    # Storing maximum value of groups for plotting
                    maxNPlt = np.max([np.nanmax(grpsREFw), np.nanmax(grpsObsw), maxNPlt])

                    print(grpsREFw.shape, grpsObsw.shape)

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
                calRef.append([])
                calObs.append([])

        # Creating storing dictionaries to store fit properties
        rSqI = []
        mResI = []
        mRResI = []

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


        for siInx in range(0, ssn_data.cenPoints['OBS'].shape[0]):

            # Analyze period only if valid
            if ssn_data.vldIntr[siInx]:

                # Calculate goodness of fit if overlap is true
                if obsRefOvrlp:

                    calRefT = calRef[siInx].copy()
                    calObsT = calObs[siInx].copy()

                    metricsDic = self._Calculate_R2M_MRes_MRRes(calObsT, calRefT, centers, edges)

                    rSqI.append(metricsDic['rSq'])
                    mResI.append(metricsDic['mRes'])
                    mRResI.append(metricsDic['mRRes'])

                else:            
            
                    rSqI.append([])
                    mResI.append([])
                    mRResI.append([])
                  

            # If period not valid store an empty array
            else:
                rSqI.append([])
                mResI.append([])
                mRResI.append([])

        rSqDT = np.nan
        mResDT = np.nan
        mRResDT = np.nan

        # Only if there is only one interval that is valid
        if len(calRef) > 0:

            calRefT = np.concatenate(calRef, axis=0)
            calObsT = np.concatenate(calObs, axis=0)

            metricsDic = self._Calculate_R2M_MRes_MRRes(calObsT, calRefT, centers, edges)

            rSqDT = metricsDic['rSq']
            mResDT = metricsDic['mRes']
            mRResDT = metricsDic['mRRes']


        # Storing variables in object-----------------------------------------------------------------------------------
#         ssn_data.LowALlim = LowALlim # Data to obtain the coefficients for the fits of the low solar activity
#         ssn_data.HighALlim = HighALlim # Data to obtain the coefficients for the fits of the high solar activity
        
#         ssn_data.xlow = xlow # Data of the x-axis for the fit to obtain the threshold for low solar activity
#         ssn_data.ylow = ylow # Data of the y-axis for the fit to obtain the threshold for low solar activity        
        
#         ssn_data.a1high = a1high # Coefficient #1 of the fit for high solar activity
#         ssn_data.a0high = a0high # Coefficient #0 of the fit for high solar activity
#         ssn_data.a1low = a1low # Coefficient #1 of the fit for low solar activity
#         ssn_data.a0low = a0low # Coefficient #0 of the fit for low solar activity   
        ssn_data.lowth = lowth # Threshold for low solar activity
        ssn_data.highth = highth # Threshold for high solar activity

        ssn_data.GDObsI = GDObsI  # Variable that stores the number of days with groups of the observer for each interval, threshold, window shift, and window
        ssn_data.ODObsI = ODObsI  # Variable that stores the number of days with observations of the observer for each interval, threshold, window shift, and window
        ssn_data.QDObsI = QDObsI  # Variable that stores the number of quiet days of the observer for each interval, threshold, window shift, and window
        ssn_data.GDREFI = GDREFI  # Variable that stores the number of days with groups of the reference for each interval, threshold, window shift, and window
        ssn_data.ODREFI = ODREFI  # Variable that stores the number of days with observations of the reference for each interval, threshold, window shift, and window
        ssn_data.QDREFI = QDREFI  # Variable that stores the number of quiet days of the reference for each interval, threshold, window shift, and window
        
        ssn_data.SNdObsI = SNdObsI  # Variable that stores the daily sunspot number of days for each interval, threshold, window shift, and window
        ssn_data.SNdREFI = SNdREFI  # Variable that stores the daily sunspot number of days for each interval, threshold, window shift, and window

        ssn_data.EMDD = EMDD  # Variable that stores the EMD between the reference and the observer for each interval, threshold, and window shift
        ssn_data.EMDtD = EMDtD  # Variable that stores the windowshift matching EMDD for each interval, threshold, and window shift
        ssn_data.EMDthD = EMDthD  # Variable that stores the threshold matching EMDD for each interval, threshold, and window shift

        ssn_data.Dis = Dis  # Distance matrix used to calcualte the EMD
        
        ssn_data.bestTh = bestTh  # Variable that stores the nBest matches for each interval
        ssn_data.wAvI = wAvI  # Weighted threshold average based on the nBest matches for different intervals
        ssn_data.wSDI = wSDI  # Weighted threshold standard deviation based on the nBest matches for different intervals


        ssn_data.calRef = calRef  # Thresholded number of groups for reference that overlap with observer
        ssn_data.calObs = calObs  # Number of groups for observer that overlap with reference

        ssn_data.maxNPlt = maxNPlt # Maximum value of groups for plotting and calculation of standard deviations
        ssn_data.centers = centers # Centers of the bins used to plot and calculate r square
        ssn_data.edges = edges # Centers of the bins used to plot and calculate r square

        ssn_data.rSqI = rSqI  # R square of the y=x line for each separate interval
        ssn_data.mResI = mResI  # Mean residual of the y=x line for each separate interval
        ssn_data.mRResI = mRResI  # Mean relative residual of the y=x line for each separate interval

        ssn_data.rSqDT = rSqDT  # R square of the y=x line using the average threshold for each interval
        ssn_data.mResDT = mResDT  # Mean residual of the y=x line using the average threshold for each interval
        ssn_data.mRResDT = mRResDT  # Mean relative residual of the y=x line using the average threshold for each interval

        ssn_data.RiseMonths = rise_count  # Number of months in rising phase
        ssn_data.DecMonths = dec_count  # Number of months in declining phase

        # Set the simultaneous threshold to the values for the valid interval if there is only one interval
        if len(calRef) == 1:
            ssn_data.wAv = wAvI[ssn_data.vldIntr][0]
            ssn_data.wSD = wSDI[ssn_data.vldIntr][0]
            ssn_data.rSqOO = rSqDT
            ssn_data.mResOO = mResDT
            ssn_data.mRResOO = mRResDT

             # Determine which threshold to use
            Th = ssn_data.wAvI[ssn_data.vldIntr][0]

            # Calculating number of groups in reference data for given threshold
            grpsREFw = np.nansum(np.greater(ssn_data.REF_Dat.values[:, 3:ssn_data.REF_Dat.values.shape[1] - 3], Th),
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

            metricsDic = self._Calculate_R2M_MRes_MRRes(grpsObsw, grpsREFw, centers, edges)

            ssn_data.rSq = metricsDic['rSq']
            ssn_data.mRes = metricsDic['mRes']
            ssn_data.mRRes = metricsDic['mRRes']


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


    def _disThres_Limit(self,
                        ssn_data,
                        disThres
                        ):
        """
        Internal function that Function that identifies valid indices for each interval that are below a relative threshold

        :param disThres: Threshold above which we will ignore timeshifts (in units of the shortest
                         distance between observer and reference ADFs for each sub-interval separately)
        :returns valShfInx, valShfLen:  Variables  with the indices and lengths to calculate the number of permutations
        """
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

        return valShfInx, valShfLen


    def ADFsimultaneousEMD(self,
                           ssn_data,
                           disThres=3,
                           MaxIter=2000):

        """
        Function that peforms the EMD optimization by allowing variations of shift while keeping thresholds constant
        VARIABLES APPENDED TO THE OBJECT ARE SPECIFIED AT THE END

        :param disThres: Threshold above which we will ignore timeshifts (in units of the shortest
                         distance between observer and reference ADFs for each sub-interval separately)
        :param MaxIter:  Maximum number of iterations above which we skip simultaneous fit
        """

        print('Identify valid shifts for simultaneous fitting...', flush=True)

        valShfInx, valShfLen = self._disThres_Limit(ssn_data, disThres)

        # Perform binary search to get as close as possible to the limit
        if np.nanprod(valShfLen) > MaxIter:

            disThresT = disThres * 0.5
            for n in range(2, 30):

                valShfInx, valShfLen = self._disThres_Limit(ssn_data, disThresT)

                if np.nanprod(valShfLen) > MaxIter:
                    disThresT = disThresT - disThres / np.power(2, n)
                else:
                    disThresT = disThresT + disThres / np.power(2, n)

            valShfInx, valShfLen = self._disThres_Limit(ssn_data, disThresT)


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
            ssn_data.mRRes = np.nan  # Mean relative residual of the y=x line using a common threshold

            ssn_data.rSqOO = np.nan  # R square of the y=x line using a common threshold, but only the valid intervals
            ssn_data.mResOO = np.nan  # Mean residual of the y=x line using a common threshold, but only the valid intervals
            ssn_data.mRResOO = np.nan  # Mean relative residual of the y=x line using a common threshold, but only the valid intervals

            return False

        print('Optimize EMD by varying shifts, but using the same threshold...', flush=True)

        # Allocating variable to store top matches
        EMDComb = np.ones((ssn_data.cenPoints['OBS'].shape[0] + 2, config.NBEST)) * 10000

        print('EMDComb',EMDComb.shape)

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
                        
                        # Numerator and denominator for given observer
                        numADObsII = ssn_data.GDObsI[siInx][TIdx,valShfInx[siInx][comb[siInx]],ssn_data.ODObsI[siInx][TIdx,valShfInx[siInx][comb[siInx]],:]/ssn_data.MoLngt>=ssn_data.minObD]
                        numQDObsII = ssn_data.MoLngt - ssn_data.QDObsI[siInx][TIdx,valShfInx[siInx][comb[siInx]],ssn_data.ODObsI[siInx][TIdx,valShfInx[siInx][comb[siInx]],:]/ssn_data.MoLngt>=ssn_data.minObD] 
                        denFMObsII = ssn_data.GDObsI[siInx][TIdx,valShfInx[siInx][comb[siInx]],ssn_data.ODObsI[siInx][TIdx,valShfInx[siInx][comb[siInx]],:]/ssn_data.MoLngt>=ssn_data.minObD]*0 + ssn_data.MoLngt
                        denODObsII = ssn_data.ODObsI[siInx][TIdx,valShfInx[siInx][comb[siInx]],ssn_data.ODObsI[siInx][TIdx,valShfInx[siInx][comb[siInx]],:]/ssn_data.MoLngt>=ssn_data.minObD]
                        # Numerator and denominator for reference
                        numADREFII = ssn_data.GDREFI[siInx][TIdx,valShfInx[siInx][comb[siInx]],ssn_data.ODREFI[siInx][TIdx,valShfInx[siInx][comb[siInx]],:]/ssn_data.MoLngt>=ssn_data.minObD]
                        numQDREFII = ssn_data.MoLngt - ssn_data.QDREFI[siInx][TIdx,valShfInx[siInx][comb[siInx]],ssn_data.ODREFI[siInx][TIdx,valShfInx[siInx][comb[siInx]],:]/ssn_data.MoLngt>=ssn_data.minObD]
                        denFMREFII = ssn_data.GDREFI[siInx][TIdx,valShfInx[siInx][comb[siInx]],ssn_data.ODREFI[siInx][TIdx,valShfInx[siInx][comb[siInx]],:]/ssn_data.MoLngt>=ssn_data.minObD]*0 + ssn_data.MoLngt
                        denODREFII = ssn_data.ODREFI[siInx][TIdx,valShfInx[siInx][comb[siInx]],ssn_data.ODREFI[siInx][TIdx,valShfInx[siInx][comb[siInx]],:]/ssn_data.MoLngt>=ssn_data.minObD] 
                        
                        if config.NUM_TYPE == "ADF": 
                            numObsII = numADObsII
                            numREFII = numADREFII                               
                        else: 
                            numObsII = numQDObsII
                            numREFII = numQDREFII
                                
                        if config.DEN_TYPE == "OBS":
                            denObsII = denODObsII
                            denREFII = denODREFII                            
                        else:
                            denObsII = denFMObsII
                            denREFII = denFMREFII

                            
                        if config.DEN_TYPE == "DTh":
                            #defining solar activity level                            
                            MMObsII=np.logical_and((ssn_data.SNdObsI[siInx][TIdx,valShfInx[siInx][comb[siInx]],ssn_data.ODObsI[siInx][TIdx,valShfInx[siInx][comb[siInx]],:]/ssn_data.MoLngt>=ssn_data.minObD]>ssn_data.lowth), (ssn_data.SNdObsI[siInx][TIdx,valShfInx[siInx][comb[siInx]],ssn_data.ODObsI[siInx][TIdx,valShfInx[siInx][comb[siInx]],:]/ssn_data.MoLngt>=ssn_data.minObD]<ssn_data.highth))
                            MMREFII=np.logical_and((ssn_data.SNdREFI[siInx][TIdx,valShfInx[siInx][comb[siInx]],ssn_data.ODREFI[siInx][TIdx,valShfInx[siInx][comb[siInx]],:]/ssn_data.MoLngt>=ssn_data.minObD]>ssn_data.lowth), (ssn_data.SNdREFI[siInx][TIdx,valShfInx[siInx][comb[siInx]],ssn_data.ODREFI[siInx][TIdx,valShfInx[siInx][comb[siInx]],:]/ssn_data.MoLngt>=ssn_data.minObD]<ssn_data.highth))

                            HMObsII=(ssn_data.SNdObsI[siInx][TIdx,valShfInx[siInx][comb[siInx]],ssn_data.ODObsI[siInx][TIdx,valShfInx[siInx][comb[siInx]],:]/ssn_data.MoLngt>=ssn_data.minObD]>=ssn_data.highth)
                            HMREFII=(ssn_data.SNdREFI[siInx][TIdx,valShfInx[siInx][comb[siInx]],ssn_data.ODREFI[siInx][TIdx,valShfInx[siInx][comb[siInx]],:]/ssn_data.MoLngt>=ssn_data.minObD]>=ssn_data.highth)  

                            # Default numerators and denominators
                            numObsII = numADObsII
                            numREFII = numADREFII
                            denObsII = denFMObsII
                            denREFII = denFMREFII


                            numObsII[HMObsII] = numQDObsII[HMObsII]
                            numREFII[HMREFII] = numQDREFII[HMREFII]

                            denObsII[MMObsII] = denODObsII[MMObsII]
                            denREFII[MMREFII] = denODREFII[MMREFII]
                    

                        # ADF calculations
                        ADF_Obs_fracII  = np.divide(numObsII, denObsII)
                        ADF_REF_fracII  = np.divide(numREFII, denREFII)                            
                        
                        
                        
                        # If it is the first interval re-create the arrays
                        if ADFObsI.shape[0] == 0:  
                            ADFObsI = ADF_Obs_fracII                    
                            ADFREFI = ADF_REF_fracII

                        # If not, append ADF from all sub-interval for the specified shifts
                        else:
                            ADFObsI = np.append(ADFObsI,ADF_Obs_fracII)
                            ADFREFI = np.append(ADFREFI, ADF_REF_fracII)   

                # Calculating Earth Mover's Distance
                ADFObs, bins = np.histogram(ADFObsI, bins=(np.arange(0, ssn_data.MoLngt + 2) - 0.5) / ssn_data.MoLngt,
                                            density=True)
                ADFREF, bins = np.histogram(ADFREFI, bins=(np.arange(0, ssn_data.MoLngt + 2) - 0.5) / ssn_data.MoLngt,
                                            density=True)
                tmpEMD = emd(ADFREF.astype(np.float64), ADFObs.astype(np.float64), ssn_data.Dis.astype(np.float64))

                if np.any(EMDComb[0, :] > tmpEMD):

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

                    # Determining index for insertion
                    insInx = config.NBEST - np.sum(EMDComb[0, :] >= tmpEMD)

                    # Insert values
                    EMDComb = np.insert(EMDComb, insInx, insArr, axis=1)

                    # Remove last element
                    EMDComb = EMDComb[:, 0:config.NBEST ]                   

        print('done.', flush=True)
        print(' ', flush=True)

        print('Calculating average threshold and its standard deviation...', end="", flush=True)
        
        # Only plot if using more than one theshold
        if config.NBEST == 1:

            wAv = EMDComb[1, :][0] * ssn_data.thI
            wSD = np.nan

        else:
            # Constructing weights
            alph = 1-(EMDComb[0,:]-np.min(EMDComb[0,:]))/(np.max(EMDComb[0,:])-np.min(EMDComb[0,:]))

            # Weighted average
            wAv = np.sum(np.multiply(alph,EMDComb[1,:]))/np.sum(alph)

            # Weighted Standard Deviation
            wSD = np.sqrt(np.sum(np.multiply(alph,np.power(EMDComb[1,:]-wAv,2)))/np.sum(alph))        
          
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
            metricsDic = self._Calculate_R2M_MRes_MRRes(grpsObsw, grpsREFw, ssn_data.centers, ssn_data.edges)

            rSq = metricsDic['rSq']
            mRes = metricsDic['mRes']
            mRRes = metricsDic['mRRes']

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
            metricsDic = self._Calculate_R2M_MRes_MRRes(grpsObsw, grpsREFw, ssn_data.centers, ssn_data.edges)

            rSqOO = metricsDic['rSq']
            mResOO = metricsDic['mRes']
            mRResOO = metricsDic['mRRes']

        print('done.', flush=True)
        print(' ', flush=True)

        # Storing variables in object-----------------------------------------------------------------------------------

        ssn_data.disThres = disThres  # Threshold above which we will ignore timeshifts
        ssn_data.EMDComb = EMDComb  # Variable storing best simultaneous fits

        ssn_data.wAv = wAv  # Weighted threshold average based on the nBest matches for all simultaneous fits
        ssn_data.wSD = wSD  # Weighted threshold standard deviation based on the nBest matches for all simultaneous fits

        ssn_data.rSq = rSq  # R square of the y=x line using a common threshold
        ssn_data.mRes = mRes  # Mean residual of the y=x line using a common threshold
        ssn_data.mRRes = mRRes  # Mean relative residual of the y=x line using a common threshold

        ssn_data.rSqOO = rSqOO  # R square of the y=x line using a common threshold, but only the valid intervals
        ssn_data.mResOO = mResOO  # Mean residual of the y=x line using a common threshold, but only the valid intervals
        ssn_data.mRResOO = mRResOO  # Mean relative residual of the y=x line using a common threshold, but only the valid intervals

        # --------------------------------------------------------------------------------------------------------------

        print('done.', flush=True)
        print(' ', flush=True)

        return True
