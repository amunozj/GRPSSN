import csv

import numpy as np

from SSN_ADF import ssnADF
from SSN_Config import SSN_ADF_Config
import SSN_ADF_Plotter
import argparse

parser = argparse.ArgumentParser(description="Specify arguments for SSN/ADF config")
parser.add_argument('-q',"--QDF", action='store_true')
parser.add_argument('-a',"--ADF", action='store_true')
parser.add_argument('-m',"--month", action='store_true')
parser.add_argument('-o',"--obs", action='store_true')
args, leftovers = parser.parse_known_args()

#################
# CONFIGURATION #
#################

# Observer ID range and who to skip
SSN_ADF_Config.OBS_START_ID = 356
SSN_ADF_Config.OBS_END_ID = 600
SSN_ADF_Config.SKIP_OBS = [332]

# Quantity to use in the numerator of the ADF:  Active days or 1-quiet days
if args.QDF is not None:
    SSN_ADF_Config.ADF_TYPE = "QDF"   # Set to 'QDF' to use 1-QDF calculation.

if args.ADF is not None:
    SSN_ADF_Config.ADF_TYPE = "ADF"   # Set to 'ADF'  to use ADF calculation.


# Quantity to use in the denominator:  Observed days or the full month
if args.month is not None:
    SSN_ADF_Config.MONTH_TYPE = "FULLM"  # Set to 'FULLM' to use full month length to determine ADF

if args.obs is not None:
    SSN_ADF_Config.MONTH_TYPE = "OBS"  # Set to 'OBS' to use observed days to determine ADF


# Flag to turn on saving of figures
plotSwitch = True

# Output Folder
output_path = 'TestFrag'

#################

# Read Data and plot reference search windows, minima and maxima
ssn_adf = ssnADF(ref_data_path='../input_data/SC_SP_RG_DB_KM_group_areas_by_day.csv',
                                  silso_path='../input_data/SN_m_tot_V2.0.csv',
                                  obs_data_path='../input_data/GNObservations_JV_V1.22.csv',
                                  obs_observer_path='../input_data/GNObservers_JV_V1.22.csv',
                                  output_path='output/' + output_path,
                                  font={'family': 'sans-serif',
                                        'weight': 'normal',
                                        'size': 21},
                                  dt=14,  # Temporal Stride in days
                                  phTol=2,  # Cycle phase tolerance in years
                                  thN=100,  # Number of thresholds including 0
                                  thI=1,  # Threshold increments
                                  plot=plotSwitch)

# Stores SSN metadata set in a SSN_ADF_Class
ssn_data = ssn_adf.ssn_data

# Creating Variable to save csv
Y_vals = []

# Naming Columns
y_row = ['Observer',
         'Station',
         'AvThreshold',
         'SDThreshold',
         'R2',
         'Avg.Res',
         'AvThresholdS',
         'SDThresholdS',
         'R2S',
         'Avg.Res.S',
         'R2DT',
         'Avg.ResDT',
         'R2OO',
         'Avg.ResOO']

Y_vals.append(y_row)

# Defining Observer
for CalObs in range(SSN_ADF_Config.OBS_START_ID, SSN_ADF_Config.OBS_END_ID):

    if CalObs in SSN_ADF_Config.SKIP_OBS:
        continue

    print("######## Beginning run on observer {} ########\n".format(CalObs))

    # Processing observer
    obs_valid = ssn_adf.processObserver(ssn_data,  # SSN metadata
                                        CalObs=CalObs,  # Observer identifier denoting observer to be processed
                                        MoLngt=30,  # Duration of the interval ("month") used to calculate the ADF
                                        minObD=0.33,
                                        # Minimum proportion of days with observation for a "month" to be considered valid
                                        vldIntThr=0.33)  # Minimum proportion of valid "months" for a decaying or raising interval to be considered valid

    # TODO: Remove obs_valid. SSN_data not being assigned before returning in SSN_ADF.py. Line 298~
    # Plot active vs. observed days
    if plotSwitch and SSN_ADF_Config.PLOT_ACTIVE_OBSERVED:
        SSN_ADF_Plotter.plotActiveVsObserved(ssn_data)

    # Continue only if observer has valid intervals
    if obs_valid:

        # Calculating the Earth's Mover Distance using sliding windows for different intervals
        obs_ref_overlap = ssn_adf.ADFscanningWindowEMD(ssn_data, nBest=50)  # Number of top best matches to keep

        if plotSwitch:
            # Plot active vs. observed days
            if SSN_ADF_Config.PLOT_OPTIMAL_THRESH:
                SSN_ADF_Plotter.plotOptimalThresholdWindow(ssn_data)

            # Plot Distribution of active thresholds
            if SSN_ADF_Config.PLOT_DIST_THRESH_MI:
                SSN_ADF_Plotter.plotDistributionOfThresholdsMI(ssn_data)

            # If there is overlap between the observer and reference plot the y=x scatterplots
            if SSN_ADF_Config.PLOT_INTERVAL_SCATTER and obs_ref_overlap and np.sum(ssn_data.vldIntr) > 1:
                SSN_ADF_Plotter.plotIntervalScatterPlots(ssn_data)

        # Calculating the Earth's Mover Distance using common thresholds for different intervals
        plot_obs = ssn_adf.ADFsimultaneousEMD(ssn_data,
                                              disThres=1.20,
                                              # Threshold above which we will ignore timeshifts in simultaneous fit
                                              MaxIter=1000)
        # Maximum number of iterations above which we skip simultaneous fit

        if plotSwitch and plot_obs:

            if np.sum(ssn_data.vldIntr) > 1:
                # Plotting minimum EMD figure
                if SSN_ADF_Config.PLOT_MIN_EMD:
                    SSN_ADF_Plotter.plotMinEMD(ssn_data)

                # Plot the result of simultaneous fit
                if SSN_ADF_Config.PLOT_SIM_FIT:
                    SSN_ADF_Plotter.plotSimultaneousFit(ssn_data)

                # Plot the distribution of thresholds
                if SSN_ADF_Config.PLOT_DIST_THRESH:
                    SSN_ADF_Plotter.plotDistributionOfThresholds(ssn_data)

            # If there is overlap between the observer and reference plot the y=x scatterplots
            if obs_ref_overlap:
                if SSN_ADF_Config.PLOT_SINGLE_THRESH_SCATTER:
                    SSN_ADF_Plotter.plotSingleThresholdScatterPlot(ssn_data)
                if SSN_ADF_Config.PLOT_MULTI_THRESH_SCATTER:
                    SSN_ADF_Plotter.plotMultiThresholdScatterPlot(ssn_data)

        # Saving row
        y_row = [ssn_data.CalObs,
                 ssn_data.NamObs,
                 ssn_data.wAv,
                 ssn_data.wSD,
                 ssn_data.rSq,
                 ssn_data.mRes,
                 ssn_data.wAvI,
                 ssn_data.wSDI,
                 ssn_data.rSqI,
                 ssn_data.mResI,
                 ssn_data.rSqDT,
                 ssn_data.mResDT,
                 ssn_data.rSqOO,
                 ssn_data.mResOO]

        Y_vals.append(y_row)

    writer = csv.writer(open('output/{}/{}Observer_ADF.csv'.format(output_path, SSN_ADF_Config.get_file_prepend(SSN_ADF_Config.ADF_TYPE, SSN_ADF_Config.MONTH_TYPE)), 'w', newline=''))
    writer.writerows(Y_vals)
