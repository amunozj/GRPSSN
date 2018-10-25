import os


class SSN_Data:
    """
    Blank class for managing SSN data.
    Can assign named variables to this class to avoid global class variables
    """
    pass


class SSN_ADF_Config:
    """
    Class to store static config variables
    """

    # MULTIPROCESSING
    # 1 --> do not use any parallel processing.
    # -1 -->  use all cores on machine.
    # Other --> defines number of cores to use
    PROCESSES = 1

    # SCATTER PLOT AND R2 OPTIONS
    # "true" --> Use sqrt(GN + 1)
    # "False" --> Use GN
    SQRT_2DHIS = False

    # OPTIMIZATION OPTIONS
    # '1' uses the absolute best combination of time and threshold and prevents the code from plotting the
    # EMD vs. Threshold plots
    # '>1' calculates the threshold using the average of the NBEST points
    NBEST = 1

    # DYNAMIC ADF OPTIONS
    # PCTLO defines the percentile of ADF months for which low activity conditions must hold
    # PCTHI defines the percentile of ADF months for which high activity conditions must hold
    PCTLO = 100
    PCTHI = 75
    # QTADF maximum ADF used to consider an interval quiet
    # ACADF minimum ADF used to consider an interval active
    QTADF = 0
    ACADF = 0.9

    # OVERWRITING AND SKIPPING PLOTS
    # Setting both flags to false will recreate and overwrite all plots for all observers
    # Overwrite plots already present
    # Even when false, still have to process each observer
    # Safer than the SKIP_OBSERVERS_WITH_PLOTS flag
    SKIP_PRESENT_PLOTS = False
    # Ignore any observer that has any plots with current flags in its output folder
    # Skips processing observer data making the process much faster
    # However, if a plot that should have been made previously is missing it will not be made when this flag is enabled
    # More dangerous than SKIP_PRESENT_PLOTS, but good when confident that existing observers were completely processed
    SKIP_OBSERVERS_WITH_PLOTS = False

    # Plotting config variables
    PLOT_SN_ADF = True
    PLOT_SN_AL = True
    PLOT_OPTIMAL_THRESH = True
    PLOT_ACTIVE_OBSERVED = True
    PLOT_DIST_THRESH_MI = True
    PLOT_INTERVAL_SCATTER = True
    PLOT_INTERVAL_DISTRIBUTION = True
    PLOT_MIN_EMD = True
    PLOT_SIM_FIT = True
    PLOT_DIST_THRESH = True
    PLOT_SINGLE_THRESH_SCATTER = True
    PLOT_SINGLE_THRESH_DIS = True
    PLOT_MULTI_THRESH_SCATTER = True
    PLOT_SMOOTHED_SERIES = True

    # Suppress numpy warnings for cleaner console output
    SUPPRESS_NP_WARNINGS = False

    @staticmethod
    def get_file_prepend(num_type, den_type):
        """
        :param num_type: ADF parameter set in config
        :param den_type: month length parameter set in config
        :return: prepend for plots depending on ADF and month length
        """

        # Type of numerator
        if num_type == "ADF":
            prepend = "A_"
        elif num_type == "QDF":
            prepend = "Q_"
        else:
            raise ValueError(
                'Invalid flag: Use \'ADF\' (or \'QDF\') for active (quiet) day fraction')

        # Type of denominator
        if den_type == "FULLM":
            prepend += "M_"
        elif den_type == "OBS":
            prepend += "O_"
        elif den_type == "DTh":
            prepend += "D_"
        else:
            raise ValueError(
                'Invalid flag: Use \'OBS\' (or \'FULLM\') to use observed days (full month length), or use \'DTh\' for dynamic ADF.')

        prepend += "NB" + str(SSN_ADF_Config.NBEST)
        if SSN_ADF_Config.DEN_TYPE == 'DTh':
            prepend += "_PL" + str(SSN_ADF_Config.PCTLO) + "_PH" + str(SSN_ADF_Config.PCTHI)
            prepend += "_QD" + str(SSN_ADF_Config.QTADF) + "_AD" + str(SSN_ADF_Config.ACADF)

        return prepend

    @staticmethod
    def get_file_output_string(number, title, ssn_data, num_type, den_type):
        """
        :param number: Plot type identifier
        :param title: Plot title
        :param ssn_data: SSN_Data object storing metadata
        :param num_type: ADF parameter set in config
        :param den_type: month length parameter set in config
        :return: Path
        """
        return os.path.join(ssn_data.output_path,
                            "{}_{}".format(ssn_data.CalObs, ssn_data.NamObs),
                            "{}_{}_{}_{}_{}.png".format(number,
                                                        SSN_ADF_Config.get_file_prepend(num_type, den_type),
                                                        ssn_data.CalObs,
                                                        ssn_data.NamObs,
                                                        title))