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

    # ADF/QDF and FULL/OBS default values
    ADF_TYPE = "QDF"
    MONTH_TYPE = "FULLM"

    # MULTI THREADING
    PROCESSES = 1 # 1 -- do not use any parallel processing.
                # -1 --  use all cores on machine.
                # Otherwise -- defines number of cores to use.

    # Choose to overwrite observers with plots already created, or skip these observers
    OVERWRITE_OBSERVERS = False


    # Plotting config varibales
    PLOT_OPTIMAL_THRESH = True
    PLOT_ACTIVE_OBSERVED = True
    PLOT_DIST_THRESH_MI = True
    PLOT_INTERVAL_SCATTER = True
    PLOT_MIN_EMD = True
    PLOT_SIM_FIT = True
    PLOT_DIST_THRESH = True
    PLOT_SINGLE_THRESH_SCATTER = True
    PLOT_MULTI_THRESH_SCATTER = True

    @staticmethod
    def get_file_prepend(adf_type, month_type):
        if adf_type == "ADF":
            prepend = "A_"
        elif adf_type == "QDF":
            prepend = "Q_"
        else:
            raise ValueError('Invalid flag: Use \'ADF\' (or \'QDF\') for active (1-quiet) day fraction.')

        if month_type == "FULLM":
            prepend += "M_"
        elif month_type == "OBS":
            prepend += "O_"
        else:
            raise ValueError('Invalid flag: Use \'OBS\' (or \'FULLM\') to use observed days (full month length) to determine ADF.')
        return prepend

    @staticmethod
    def get_file_output_string(number, title, ssn_data, adf_type, month_type):
        return '{}/{}_{}/{}{}_{}_{}_{}.png'.format(
            ssn_data.output_path,
            ssn_data.CalObs,
            ssn_data.NamObs,
            SSN_ADF_Config.get_file_prepend(adf_type, month_type),
            number,
            ssn_data.CalObs,
            ssn_data.NamObs,
            title)
