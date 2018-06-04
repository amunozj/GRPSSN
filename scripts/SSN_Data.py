class SSN_Data:
    """
    Blank class for managing SSN data.
    Can assign named variables to this class to avoid global class variables
    """
    pass


class SSN_Config:
    """
    Class to store static config variables
    """

    # Skip observers
    SKIP_OBS = [332]

    # ADF variables
    ADF_TYPE = "ADF"  # 1 - QDF
    MONTH_TYPE = "FULL"  # ACTIVE

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

