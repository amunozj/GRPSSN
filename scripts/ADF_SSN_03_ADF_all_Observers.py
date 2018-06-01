# Modules
import sys
import os.path
import csv
import numpy as np

sys.path.insert(1, r'../functions')  # add to pythonpath

import SSN_ADF_Class

plotSwitch = True
output_path = 'TestFrag'

# Read Data and plot reference search windows, minima and maxima
ssn_adf = SSN_ADF_Class.ssnADF_cl(ref_data_path='../input_data/SC_SP_RG_DB_KM_group_areas_by_day.csv',
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

# Creating Variable to save csv
Y_vals = []

# Naming Columns
y_row = ['Observer']
y_row += ['Station']
y_row += ['AvThreshold']
y_row += ['SDThreshold']
y_row += ['R2']
y_row += ['Avg.Res']
y_row += ['AvThresholdS']
y_row += ['SDThresholdS']
y_row += ['R2S']
y_row += ['Avg.Res.S']
y_row += ['R2DT']
y_row += ['Avg.ResDT']
y_row += ['R2OO']
y_row += ['Avg.ResOO']
Y_vals.append(y_row)

# Defining Observer
for CalObs in range(1, 600):

    #CalObs = 412

    if CalObs != 332:

        # Processing observer
        obs_valid = ssn_adf.processObserver(CalObs=CalObs,  # Observer identifier denoting observer to be processed
                                            MoLngt=30,  # Duration of the interval ("month") used to calculate the ADF
                                            minObD=0.33,  # Minimum proportion of days with observation for a "month" to be considered valid
                                            vldIntThr=0.33)  # Minimum proportion of valid "months" for a decaying or raising interval to be considered valid

        # Plot active vs. observed days
        if plotSwitch:
            ssn_adf.plotActiveVsObserved()

        # Continue only if observer has valid intervals
        if obs_valid:

            # Calculating the Earth's Mover Distance using sliding windows for different intervals
            obs_ref_overlap = ssn_adf.ADFscanningWindowEMD(nBest=50)  # Number of top best matches to keep

            if plotSwitch:
                # Plot active vs. observed days
                ssn_adf.plotOptimalThresholdWindow()

                # Plot Distribution of active thresholds
                ssn_adf.plotDistributionOfThresholdsMI()

                # If there is overlap between the observer and reference plot the y=x scatterplots
                if obs_ref_overlap and np.sum(ssn_adf.vldIntr) > 1:
                    ssn_adf.plotIntervalScatterPlots()



            # Calculating the Earth's Mover Distance using common thresholds for different intervals
            plot_obs = ssn_adf.ADFsimultaneousEMD(disThres=1.20,  # Threshold above which we will ignore timeshifts in simultaneous fit
                                       MaxIter=1000)  # Maximum number of iterations above which we skip simultaneous fit

            if plotSwitch and plot_obs:

                if np.sum(ssn_adf.vldIntr) > 1:
                    # Plotting minimum EMD figure
                    ssn_adf.plotMinEMD()

                    # Plot the result of simultaneous fit
                    ssn_adf.plotSimultaneousFit()

                    # Plot the distribution of thresholds
                    ssn_adf.plotDistributionOfThresholds()

                # If there is overlap between the observer and reference plot the y=x scatterplots
                if obs_ref_overlap:
                    ssn_adf.plotSingleThresholdScatterPlot()
                    ssn_adf.plotMultiThresholdScatterPlot()

            # Saving row
            y_row = [ssn_adf.CalObs]
            y_row += [ssn_adf.NamObs]
            y_row += [ssn_adf.wAv]
            y_row += [ssn_adf.wSD]
            y_row += [ssn_adf.rSq]
            y_row += [ssn_adf.mRes]
            y_row += [ssn_adf.wAvI]
            y_row += [ssn_adf.wSDI]
            y_row += [ssn_adf.rSqI]
            y_row += [ssn_adf.mResI]
            y_row += [ssn_adf.rSqDT]
            y_row += [ssn_adf.mResDT]
            y_row += [ssn_adf.rSqOO]
            y_row += [ssn_adf.mResOO]
            Y_vals.append(y_row)

        writer = csv.writer(open('output/' + output_path + '/Observer_ADF.csv', 'w', newline=''))
        writer.writerows(Y_vals)
