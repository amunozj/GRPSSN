import csv
import numpy as np
from SSN_GRP import ssnGRP
from SSN_Config import SSN_GRP_Config
import SSN_GRP_Plotter
import argparse
from multiprocessing import Pool
from SSN_Config import SSN_GRP_Config as config
import os

parser = argparse.ArgumentParser(description="Specify arguments for SSN/GRP config")
parser.add_argument('-O', "--obs", help="Run using total observed days in GRP calculation", action='store_true')
parser.add_argument("-t", "--threads", help="Number of threads to use in multiprocessing", type=int)
parser.add_argument("--start-id", help="ID of the observer to start at", type=int)
parser.add_argument("--end-id", help="ID of the observer to end at", type=int)
parser.add_argument("--skip-observers", help="Ignore observers who have a plot present "
                                             "(see SKIP_OBSERVERS_WITH_PLOTS in config)", action='store_true')
parser.add_argument("--suppress-warnings", help="Suppress all numpy related warnings (use with caution)"
                    , action='store_true')

args, leftovers = parser.parse_known_args()

#################
# CONFIGURATION #
#################

# Observer ID range and who to skip
SSN_GRP_Config.OBS_START_ID = 636
SSN_GRP_Config.OBS_END_ID = 710
SSN_GRP_Config.SKIP_OBS = [574, 579, 635]

# Flag to turn on saving of figures
plotSwitch = True

###################
# PARSING ARGUMENTS#
###################
# Arguments will over-ride the options set above

# Set number of threads
if args.threads is not None:
    SSN_GRP_Config.PROCESSES = args.threads

# Set start and end ID of observer loop through command line
if args.start_id is not None:
    SSN_GRP_Config.OBS_START_ID = args.start_id
if args.end_id is not None:
    SSN_GRP_Config.OBS_END_ID = args.end_id

# Flag to skip over already processed observers (see config file for more detail)
if args.skip_observers:
    SSN_GRP_Config.SKIP_OBSERVERS_WITH_PLOTS = args.skip_observers

# Suppress numpy warning messages
if args.suppress_warnings:
    SSN_GRP_Config.SUPPRESS_NP_WARNINGS = args.suppress_warnings

if SSN_GRP_Config.SUPPRESS_NP_WARNINGS:
    np.warnings.filterwarnings('ignore')

# Output Folder
output_path = 'MulL2'
# output_path = SSN_GRP_Config.get_file_prepend(SSN_GRP_Config.NUM_TYPE, SSN_GRP_Config.DEN_TYPE)

# Output CSV file path
output_csv_file = 'output/{}/{}_Observer1_GRP.csv'.format(output_path, SSN_GRP_Config.get_file_prepend())

#################
# STARTING SCRIPT#
#################

print(
    "Starting script \n")

# Naming Columns
header = ['Observer',
          'Station',
          'AvThreshold',  # Weighted threshold average based on the nBest matches for all simultaneous fits
          'SDThreshold',  # Weighted threshold standard deviation based on the nBest matches for all simultaneous fits
          'AvThresholdS',  # Weighted threshold average based on the nBest matches for different intervals
          'RealThreshold',  # Best threshold identified by minimizing difference between smoothed series
          'RealThresholdM',  # Best threshold identified by minimizing difference between smoothed series for each valid interval
          # Smoothed series metrics
          'mreSth',  # Mean relative error - single threshold
          'mneSth',  # Mean normalized error - single threshold
          'KSth',  # K-factor - single threshold
          'mreMth',  # Mean relative error - multi threshold
          'mneMth',  # Mean normalized error - multi threshold
          'KMth',  # Mean normalized error - multi threshold
          # Common threshold
          'R2d',  # R square
          'Mean.Res',  # Mean residual
          'Mean.Rel.Res',  # Mean relative residual
          'Slope',  # Slope of y/x
          # Separate intervals
          'R2dSI',  # R square
          'Mean.Res.SI',  # Mean residual
          'Mean.Rel.Res.SI',  # Mean relative residual
          'Slope.SI',  # Slope of y/x
          # Using the different threshold for each interval
          'R2dDifT',  # R square
          'Mean.ResDifT',  # Mean residual
          'Mean.Rel.ResDifT',  # Mean relative residual
          'Slope.DifT',  # Slope of y/x
          # Common threshold, but only the valid intervals
          'R2dVI',  # R square
          'Mean.ResVI',  # Mean residual
          'Mean.Rel.ResVI',  # Mean Relative residual
          'Slope.ResVI',  # Slope of y/x
          # Other Observing variables
          'QDays',  # Total number of Quiet days
          'ADays',  # Total number of Active days
          'NADays',  # Total number of missing days in data
          'QAFrac',  # Fraction of quiet to active days
          'RiseDays',  # Number of valid days in rising phases
          'DecDays',  # Number of valid days in declining phases
          'ObsStartDate',  # Starting date
          'ObsTotLength'  # Days between starting and ending dates
          ]

# Read Data and plot reference search windows, minima and maxima
ssn_grp = ssnGRP(ref_data_path='../input_data/SC_SP_RG_DB_KM_group_areas_by_day.csv',
                 silso_path='../input_data/SN_m_tot_V2.0.csv',
                 silso_path_daily='../input_data/SN_d_tot_V2.0.csv',
                 obs_data_path='../input_data/GNobservations_JV_V1.22.csv',
                 obs_observer_path='../input_data/GNobservers_JV_V1.22.csv',
                 output_path='output/' + output_path,
                 font={'family': 'sans-serif',
                       'weight': 'normal',
                       'size': 21},
                 minYrRef=1900,  #Minimum year used for reference
                 dt=20,  # Temporal Stride in days
                 phTol=1.0,  # Cycle phase tolerance in years
                 thS=1,  # Starting threshold
                 thE=120, # Ending Threshold
                 thI=5,  # Threshold increments
                 minObD=30,  # Minimum number of days with non-zero groups to consider an interval valid
                 maxValInt=2,  # Maximum number of valid intervals to be used in calibration
                 GssKerPlt=75,  # Number of days used on the gaussian kernel smoothing filter for plotting
                 plot=plotSwitch)

# Stores SSN metadata set in a SSN_GRP_Class
ssn_data = ssn_grp.ssn_data

if plotSwitch:
    SSN_GRP_Plotter.plotSearchWindows(ssn_data)

if not os.path.exists(output_csv_file):
    with open(output_csv_file, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(header)


# Start pipeline for an individual observer
def run_obs(CalObsID):
    if CalObsID in SSN_GRP_Config.SKIP_OBS:
        return

    print("######## Starting run on observer {} ########\n".format(CalObsID), flush=True)
    print(' ', flush=True)

    # Processing observer
    obs_valid = ssn_grp.processObserver(CalObs=CalObsID,  # Observer identifier denoting observer to be processed
                                        realThTol=0.05,  # Proportionality tolerance for calculating real threshold
                                        medianSw=True)  # Switch that determines whether the error metrics are calculated on all points (False) or the medians of binned data (True)

    # Continue only if observer has valid intervals
    if obs_valid:

        # Plot active vs. observed days
        if plotSwitch and SSN_GRP_Config.PLOT_ACTIVE_OBSERVED:
            SSN_GRP_Plotter.plotActiveVsObserved(ssn_data)

        # Calculating the Earth's Mover Distance using sliding windows for different intervals
        obs_ref_overlap = ssn_grp.GRPscanningWindowDis(noOvrlpSw=True,  # Switch that forces the code to ignore the true overlapping phase in calibration if present
                                                       emdSw=True,   # Switch that activates the EMD metric (True), vs the L2 norm (False)
                                                       onlyActiveSw=False, # Switch that ignores (True) or includes zeros (False)
                                                       Dis_Pow=1,    # Power index used to define the distance matrix for EMD calculation
                                                       NGrpsBins=26, # Number of used bins to calculate the group distribution
                                                       nBest1=100,   # Number of best matches to keep based on optimal distance
                                                       nBest2=10,    # Number of best matches to keep based on similarity in number of observations
                                                       sigmaTh=2.0,  # Sigma in threshold of the smoothing gaussian filter applied to the optimization matrix
                                                       sigmaT=15)   # Sigma in days of the smoothing gaussian filter applied to the optimization matrix

        if plotSwitch:
            # Plot active vs. observed days
            if SSN_GRP_Config.PLOT_OPTIMAL_THRESH:
                SSN_GRP_Plotter.plotOptimalThresholdWindow(ssn_data)

            # Plot Distribution of active thresholds
            if SSN_GRP_Config.PLOT_DIST_THRESH_MI:
                SSN_GRP_Plotter.plotDistributionOfThresholdsMI(ssn_data)

            # If there is overlap between the observer and reference plot the y=x scatterplots
            if SSN_GRP_Config.PLOT_INTERVAL_SCATTER and obs_ref_overlap:
                SSN_GRP_Plotter.plotIntervalScatterPlots(ssn_data)

            # Plot optimal distributions for each threshold
            if SSN_GRP_Config.PLOT_INTERVAL_DISTRIBUTION:
                SSN_GRP_Plotter.plotIntervalDistributions(ssn_data)

        # Calculating the Earth's Mover Distance using common thresholds for different intervals
        if np.sum(ssn_data.vldIntr) > 1:
            plot_EMD_obs = ssn_grp.GRPsimultaneousEMD(nBest1m=100,  # Number of best matches to keep based on optimal distance
                                                      nBest2m=10)   # Number of best matches to keep based on similarity in number of observations


        # Calculate smoothed series for comparison
        ssn_grp.smoothedComparison(gssnKrnl=75)  # Width of the gaussian smoothing kernel in days

        if plotSwitch:

            if np.sum(ssn_data.vldIntr) > 1 and plot_EMD_obs:

                # Plot the result of simultaneous fit
                if SSN_GRP_Config.PLOT_SIM_FIT:
                    SSN_GRP_Plotter.plotSimultaneousFit(ssn_data)

                # Plot the distribution of thresholds
                if SSN_GRP_Config.PLOT_DIST_THRESH:
                    SSN_GRP_Plotter.plotDistributionOfThresholds(ssn_data)

                if SSN_GRP_Config.PLOT_MULTI_THRESH_SCATTER and obs_ref_overlap:
                    SSN_GRP_Plotter.plotMultiThresholdScatterPlot(ssn_data)

                if SSN_GRP_Config.PLOT_SINGLE_THRESH_DIS:
                    SSN_GRP_Plotter.plotSingleThresholdDistributions(ssn_data)

            # If there is overlap between the observer and reference plot the y=x scatterplots
            if obs_ref_overlap:
                if SSN_GRP_Config.PLOT_SINGLE_THRESH_SCATTER:
                    SSN_GRP_Plotter.plotSingleThresholdScatterPlot(ssn_data)

                if SSN_GRP_Config.PLOT_SMOOTHED_SERIES:
                    SSN_GRP_Plotter.plotSmoothedSeries(ssn_data)

        # Saving row
        y_row = [ssn_data.CalObs,
                 ssn_data.NamObs,
                 ssn_data.bTh,  # Weighted threshold average based on the nBest matches for all simultaneous fits
                 ssn_data.bThI,  # Weighted threshold average based on the nBest matches for different intervals
                 ssn_data.optimThS,  # Best threshold identified by minimizing difference between smoothed series
                 ssn_data.optimThM,  # Best threshold identified by minimizing difference between smoothed series for each valid interval
                 # Smoothed series metrics
                 ssn_data.mreSth,  # Mean relative error - single threshold
                 ssn_data.mneSth,  # Mean normalized error - single threshold
                 ssn_data.slpSth,  # K-factor - single threshold
                 ssn_data.mreMth,  # Mean relative error - multi threshold
                 ssn_data.mneMth,  # Mean normalized error - multi threshold
                 ssn_data.slpMth,  # K-factor  - multi threshold
                 # Common threshold
                 ssn_data.mD['rSq'],  # R square
                 ssn_data.mD['mRes'],  # Mean residual
                 ssn_data.mD['mRRes'],  # Mean relative residual
                 ssn_data.mD['Slope'],  # Slope of y/x
                 # Separate intervals
                 ssn_data.rSqI,  # R square
                 ssn_data.mResI,  # Mean residual
                 ssn_data.mRResI,  # Mean relative residual
                 ssn_data.slopeI,  # Slope of y/x
                 # Using the different threshold for each interval
                 ssn_data.mDDT['rSq'],  # R square
                 ssn_data.mDDT['mRes'],  # Mean residual
                 ssn_data.mDDT['mRRes'],  # Mean relative residual
                 ssn_data.mDDT['Slope'],  # Slope of y/x
                 # Common threshold, but only the valid intervals
                 ssn_data.mDOO['rSq'],  # R square
                 ssn_data.mDOO['mRes'],  # Mean residual
                 ssn_data.mDOO['mRRes'],  # Mean relative residual\
                 ssn_data.mDOO['Slope'],  # Slope of y/x
                 # Other Observing variables
                 ssn_data.QDays,  # Total number of Quiet days
                 ssn_data.ADays,  # Total number of Active days
                 ssn_data.NADays,  # Total number of missing days in data
                 ssn_data.QAFrac,  # Fraction of quiet to active days
                 ssn_data.RiseDays,  # Number of valid days in rising phases
                 ssn_data.DecDays,  # Number of valid days in declining phases
                 ssn_data.ObsStartDate,  # Start date
                 ssn_data.ObsTotLength # Days between starting and ending dates
                 ]

        with open(output_csv_file, 'a+', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(y_row)


if __name__ == '__main__':

    obs_range = range(SSN_GRP_Config.OBS_START_ID, SSN_GRP_Config.OBS_END_ID)

    # If SKIP_OBSERVERS_WITH_PLOTS is set, for each observer find matching directory
    # If there is a file with the same flags as currently set, add this observer to list of observers to skip
    if SSN_GRP_Config.SKIP_OBSERVERS_WITH_PLOTS:
        out_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'output', output_path)
        for ob in obs_range:
            for dname in os.listdir(out_dir):
                if dname.startswith('{}_'.format(ob)):
                    for file in os.listdir(os.path.join(out_dir, dname)):
                        if file.startswith(
                                SSN_GRP_Config.get_file_prepend(SSN_GRP_Config.GRP_TYPE,
                                                                SSN_GRP_Config.MONTH_TYPE)):
                            SSN_GRP_Config.SKIP_OBS.append(ob)
                            break
        print("\nSkipping observers who have plot(s) labeled with the current flags ({} / {}):\n{}\n"
              "Change the SKIP_OBSERVERS_WITH_PLOTS config flag to remove this behavior\n".format(
            SSN_GRP_Config.GRP_TYPE, SSN_GRP_Config.MONTH_TYPE, SSN_GRP_Config.SKIP_OBS))

    if SSN_GRP_Config.PROCESSES == 1:
        for i in obs_range:
            run_obs(i)  # Start process normally
    elif SSN_GRP_Config.PROCESSES == -1:
        try:
            pool = Pool()  # Create a multiprocessing Pool with all available cores
            pool.map(run_obs, obs_range)  # process all observer iterable with pool
        finally:  # To make sure processes are closed in the end, even if errors happen
            pool.close()
            pool.join()
    else:
        # Safety checks for process number
        if SSN_GRP_Config.PROCESSES < -1 or SSN_GRP_Config.PROCESSES == 0:
            raise ValueError(
                "Invalid processes number ({}). Please set to a valid number.".format(SSN_GRP_Config.PROCESSES))
        elif SSN_GRP_Config.PROCESSES > os.cpu_count():
            raise ValueError("Processes number higher than CPU count. "
                             "You tried to initiate {} processes, while only having {} CPU's.".format(
                SSN_GRP_Config.PROCESSES, os.cpu_count()))
        try:
            pool = Pool(processes=SSN_GRP_Config.PROCESSES)  # Create a multiprocessing Pool
            pool.map(run_obs, obs_range)  # process all observer iterable with pool
        finally:  # To make sure processes are closed in the end, even if errors happen
            pool.close()
            pool.join()
