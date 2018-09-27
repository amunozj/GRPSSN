import numpy as np
import scipy as sp
from astropy import convolution as conv
import matplotlib.pyplot as plt
from matplotlib import colors as clrs
from pyemd import emd
from SSN_Config import SSN_ADF_Config as config
import os


def plotSearchWindows(ssn_data, SILSO_Sn, SIL_max, SIL_min, REF_min, REF_max,
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

    figure_path = '{}/01_Search_Windows.png'.format(ssn_data.output_path)

    print('Creating and saving search window figure...', end="", flush=True)

    font = ssn_data.font
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
    ax1.plot(SILSO_Sn['FRACYEAR'], SILSO_Sn['MMEAN'], color=ssn_data.Clr[0], linewidth=2)
    ax1.plot(SILSO_Sn['FRACYEAR'], SILSO_Sn['MSMOOTH'], color=ssn_data.Clr[3], linewidth=4)
    ax1.scatter(SIL_max['FRACYEAR'], SIL_max['MSMOOTH'], color='r', edgecolor='r', alpha=1, s=100, linewidths=2,
                zorder=10)
    ax1.scatter(SIL_min['FRACYEAR'], SIL_min['MSMOOTH'], color='b', edgecolor='b', alpha=1, s=100, linewidths=2,
                zorder=10)
    ax1.scatter(REF_min['FRACYEAR'], REF_min['MSMOOTH'], color='none', edgecolor='yellow', alpha=1, s=100, linewidths=3,
                zorder=10)
    ax1.scatter(REF_max['FRACYEAR'], REF_max['MSMOOTH'], color='none', edgecolor='yellow', alpha=1, s=100, linewidths=3,
                zorder=10)
    ax1.fill(ssn_data.REF_Dat['FRACYEAR'], ssn_data.risMask['PLOT'] * np.max(SILSO_Sn['MMEAN']),
             edgecolor=ssn_data.Clr[4],
             color=ssn_data.Clr[4], alpha=0.3,
             zorder=15)
    ax1.fill(ssn_data.REF_Dat['FRACYEAR'], ssn_data.decMask['PLOT'] * np.max(SILSO_Sn['MMEAN']),
             edgecolor=ssn_data.Clr[2],
             color=ssn_data.Clr[2], alpha=0.3,
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


def plotActiveVsObserved(ssn_data,
                         dpi=300,
                         pxx=4000,
                         pxy=1300,
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

    print('Creating and saving active vs. observed days figure...', end="", flush=True)

    figure_path = config.get_file_output_string('02', 'active_vs_observed_days',
                                                ssn_data=ssn_data,
                                                num_type=config.NUM_TYPE,
                                                den_type=config.DEN_TYPE)

    if config.SKIP_PRESENT_PLOTS and os.path.exists(figure_path):
        print(
            "\nFigure at {} already exists.\n Change the SKIP_PRESENT_PLOTS config flag to overwrite existing plots\n".format(
                figure_path).format(figure_path))
        return

    # Selecting the maximum integer amount of "months" out of the original data
    grpsOb = ssn_data.ObsDat['GROUPS'].values
    grpsOb = grpsOb[0:np.int(grpsOb.shape[0] / ssn_data.MoLngt) * ssn_data.MoLngt]

    ordOb = ssn_data.ObsDat['ORDINAL'].values
    ordOb = ordOb[0:np.int(ordOb.shape[0] / ssn_data.MoLngt) * ssn_data.MoLngt]

    yrOb = ssn_data.ObsDat['FRACYEAR'].values
    yrOb = yrOb[0:np.int(yrOb.shape[0] / ssn_data.MoLngt) * ssn_data.MoLngt]

    # Reshaping
    grpsOb = grpsOb.reshape((-1, ssn_data.MoLngt))
    ordOb = ordOb.reshape((-1, ssn_data.MoLngt))
    yrOb = yrOb.reshape((-1, ssn_data.MoLngt))

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

    font = ssn_data.font
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
    ax1.fill(pltxOb, pltyOb, color=ssn_data.Clr[2])
    ax1.fill(pltxOb, pltyGr, color=ssn_data.Clr[4])
    # Add number of days with groups (not including zeros and days without observations)

    ax1.plot(np.array([np.min(pltxOb), np.max(pltxOb)]), np.array([1, 1]) * ssn_data.minObD * ssn_data.MoLngt, 'k--')

    # Axes properties
    ax1.text(0.5, 1.14, 'Comparison of active vs. observed days for ' + ssn_data.NamObs,
             horizontalalignment='center',
             fontsize=20,
             transform=ax1.transAxes)
    ax1.set_ylabel('Number of days')
    ax1.legend(['Required Minimum of Observed Days', 'Observed days', 'Active days'], loc='upper right', ncol=3,
               frameon=True, edgecolor='none')
    ax1.set_xlim(left=np.min(fyr1Ob), right=np.max(fyr2Ob))
    ax1.set_ylim(bottom=0.01 * ssn_data.MoLngt, top=1.19 * ssn_data.MoLngt)
    ax1.grid(color=(0.5, 0.5, 0.5), linestyle='--', linewidth=1, axis='x', which='both')
    ax1.xaxis.tick_top()
    ax1.minorticks_on()

    # Active/observation/missing mesh
    AcObMesh = np.isfinite(grpsOb).astype(int) + np.greater(grpsOb, 0).astype(int)
    xMesh = np.insert(fyr2Ob, 0, fyr1Ob[0])
    yMesh = np.arange(0, ssn_data.MoLngt + 1)

    # Colormap
    colors = [(1, 1, 1), ssn_data.Clr[2], ssn_data.Clr[4]]
    cmap = clrs.LinearSegmentedColormap.from_list('cmap', colors, N=3)

    ax2 = fig.add_axes([ppadh, ppadv + pxy / fszv, pxx / fszh, pxy / fszv], sharex=axd)
    ax2.pcolormesh(xMesh, yMesh, np.transpose(AcObMesh), cmap=cmap, alpha=0.3, linewidth=2)
    ax2.set_ylim(bottom=0.1, top=ssn_data.MoLngt)

    # Axes properties
    ax2.set_ylabel('Day of the month')
    ax2.grid(color=(0.5, 0.5, 0.5), linestyle='--', linewidth=1, axis='x', which='both')
    ax2.minorticks_on()

    # Average group number
    ax3 = fig.add_axes([ppadh, ppadv, pxx / fszh, pxy / fszv])

    for Idx in range(0, ssn_data.cenPoints['OBS'].shape[0]):
        if ssn_data.vldIntr[Idx]:
            ax3.fill([ssn_data.endPoints['OBS'][Idx, 0], ssn_data.endPoints['OBS'][Idx, 0],
                      ssn_data.endPoints['OBS'][Idx + 1, 0],
                      ssn_data.endPoints['OBS'][Idx + 1, 0]],
                     [0, np.ceil(np.nanmax(AvGrpOb)) + 1, np.ceil(np.nanmax(AvGrpOb)) + 1, 0],
                     color=ssn_data.Clr[1 + np.mod(Idx, 2) * 2], alpha=0.2)

    ax3.plot((fyr1Ob + fyr2Ob) / 2, AvGrpOb, color=ssn_data.Clr[0], linewidth=2)

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


def plotOptimalThresholdWindow(ssn_data,
                               dpi=300,
                               pxx=4000,
                               pxy=1000,
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

    font = ssn_data.font
    plt.rc('font', **font)

    figure_path = config.get_file_output_string('03', 'Optimal_Threshold_Window',
                                                ssn_data=ssn_data,
                                                num_type=config.NUM_TYPE,
                                                den_type=config.DEN_TYPE)

    if config.SKIP_PRESENT_PLOTS and os.path.exists(figure_path):
        print(
            "\nFigure at {} already exists.\n Change the OVERWRITE_OBSERVERS config flag to overwrite existing plots\n".format(
                figure_path).format(figure_path))
        return

    print('Creating and saving optimal threshold figure...', end="", flush=True)

    nph = 1  # Number of horizontal panels
    npv = ssn_data.cenPoints['OBS'].shape[0] + 1  # Number of vertical panels

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
    ax2.plot(ssn_data.REF_Dat['FRACYEAR'], ssn_data.REF_Dat['AVGROUPS'], 'r--', linewidth=2, alpha=1)

    # Plotting Observer
    ax2.plot(ssn_data.obsPlt['X'], ssn_data.obsPlt['Y'], color=ssn_data.Clr[0], linewidth=2)

    # Axes properties
    ax2.set_ylabel('Average Number of Groups')
    ax2.set_xlabel('Center of sliding window (Year)')
    ax2.set_xlim(left=np.min(ssn_data.REF_Dat['FRACYEAR']), right=np.max(ssn_data.REF_Dat['FRACYEAR']));
    ax2.set_ylim(bottom=0, top=np.max(ssn_data.REF_Dat['AVGROUPS']) * 1.1)

    # EMD Pcolor
    plt.viridis()

    # Going through different sub-intervals
    for siInx in range(0, ssn_data.cenPoints['OBS'].shape[0]):
        # Creating axis
        ax1 = fig.add_axes([ppadh, ppadv + (siInx + 1) * (pxy / fszv + ppadv2), pxx / fszh, pxy / fszv])

        # Defining mask based on the interval type (rise or decay)
        if ssn_data.cenPoints['OBS'][siInx, 1] > 0:
            cadMaskI = ssn_data.risMask['INDEX']
            cadMask = ssn_data.risMask['PLOT']
        else:
            cadMaskI = ssn_data.decMask['INDEX']
            cadMask = ssn_data.decMask['PLOT']

        # Selecting interval
        TObsDat = ssn_data.ObsDat.loc[
            np.logical_and(ssn_data.ObsDat['FRACYEAR'] >= ssn_data.endPoints['OBS'][siInx, 0],
                           ssn_data.ObsDat['FRACYEAR'] < ssn_data.endPoints['OBS'][siInx + 1, 0])
            , 'GROUPS'].values.copy()
        TObsFYr = ssn_data.ObsDat.loc[
            np.logical_and(ssn_data.ObsDat['FRACYEAR'] >= ssn_data.endPoints['OBS'][siInx, 0],
                           ssn_data.ObsDat['FRACYEAR'] < ssn_data.endPoints['OBS'][siInx + 1, 0])
            , 'FRACYEAR'].values.copy()
        TObsOrd = ssn_data.ObsDat.loc[
            np.logical_and(ssn_data.ObsDat['FRACYEAR'] >= ssn_data.endPoints['OBS'][siInx, 0],
                           ssn_data.ObsDat['FRACYEAR'] < ssn_data.endPoints['OBS'][siInx + 1, 0])
            , 'ORDINAL'].values.copy()

        # Find index of minimum inside sub-interval
        minYear = np.min(np.absolute(TObsFYr - ssn_data.cenPoints['OBS'][siInx, 0]))
        obsMinInx = (np.absolute(TObsFYr - ssn_data.cenPoints['OBS'][siInx, 0]) == minYear).nonzero()[0][0]

        # Calculating mesh for plotting
        x = ssn_data.REF_Dat['FRACYEAR'].values[cadMaskI]
        y = np.arange(0, ssn_data.thN) * ssn_data.thI
        xx, yy = np.meshgrid(x, y)

        # Plot Matrix Only if the period is valid
        if ssn_data.vldIntr[siInx]:

            # Creating matrix for sorting and find the best combinations of threshold and shift
            OpMat = np.concatenate(
                (ssn_data.EMDtD[siInx].reshape((-1, 1)), ssn_data.EMDthD[siInx].reshape((-1, 1)),
                 ssn_data.EMDD[siInx].reshape((-1, 1))),
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
            if (TObsFYr[obsMinInx] > np.min(ssn_data.REF_Dat['FRACYEAR'])) and (
                        TObsFYr[obsMinInx] < np.max(ssn_data.REF_Dat['FRACYEAR'])):

                # Check if first element is present in reference
                if np.any(ssn_data.REF_Dat['ORDINAL'] == TObsOrd[0]):

                    # Selecting the maximum integer amount of "months" out of the original data
                    TgrpsOb = TObsDat[0:np.int(TObsDat.shape[0] / ssn_data.MoLngt) * ssn_data.MoLngt].copy()

                    # Calculating bracketing indices
                    Idx1 = (ssn_data.REF_Dat['ORDINAL'] == TObsOrd[0]).nonzero()[0][0]
                    Idx2 = Idx1 + TgrpsOb.shape[0]

                    # Going through different thresholds
                    for TIdx in range(0, ssn_data.thN):

                        # Calculating number of groups in reference data for given threshold
                        grpsREFw = np.nansum(
                            np.greater(ssn_data.REF_Dat.values[:, 3:ssn_data.REF_Dat.values.shape[1] - 3],
                                       TIdx * ssn_data.thI),
                            axis=1).astype(float)
                        grpsREFw[np.isnan(ssn_data.REF_Dat['AREA1'])] = np.nan

                        # Selecting the maximum integer amount of "months" out of the original data
                        TgrpsOb = TObsDat[0:np.int(TObsDat.shape[0] / ssn_data.MoLngt) * ssn_data.MoLngt].copy()

                        # Selecting reference window of matching size to observer sub-interval;
                        TgrpsREF = grpsREFw[Idx1:Idx2].copy()

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
                            GDObsT[ODObsT / ssn_data.MoLngt >= ssn_data.minObD],
                            ODObsT[ODObsT / ssn_data.MoLngt >= ssn_data.minObD]),
                            bins=(np.arange(0, ssn_data.MoLngt + 2) - 0.5) / ssn_data.MoLngt, density=True)

                        ADFREF, bins = np.histogram(np.divide(
                            GDREFT[ODREFT / ssn_data.MoLngt >= ssn_data.minObD],
                            ODREFT[ODREFT / ssn_data.MoLngt >= ssn_data.minObD]),
                            bins=(np.arange(0, ssn_data.MoLngt + 2) - 0.5) / ssn_data.MoLngt, density=True)

                        tmp = emd(ADFREF.astype(np.float64), ADFObs.astype(np.float64), ssn_data.Dis.astype(np.float64))

                        # Calculating Chi-Square distance
                        # ADFObs, bins = np.histogram(GDObsT[ODObsT / MoLngt >= minObD] / MoLngt,
                        #                            bins=(np.arange(0, MoLngt + 2) - 0.5) / MoLngt)
                        # ADFREF, bins = np.histogram(GDREFT[ODREFT / MoLngt >= minObD] / MoLngt,
                        #                            bins=(np.arange(0, MoLngt + 2) - 0.5) / MoLngt)

                        # Calculating numerator and denominator for Chi-square distance
                        # Nom = np.power(ADFObs - ADFREF, 2)
                        # Den = np.power(ADFObs,2) + np.power(ADFREF,2)
                        # Den = ADFObs + ADFREF

                        # Removing zeros in denominator
                        # Nom = Nom[Den != 0]
                        # Den = Den[Den != 0]

                        # Calculating Chi-square distance
                        # tmp = np.sum(np.divide(Nom, Den))

                        # Udating variables
                        if tmp < tmpEMD:
                            tmpEMD = tmp
                            tmpt = TObsFYr[obsMinInx]
                            tmpth = TIdx * ssn_data.thI

            OpMat = np.insert(OpMat, 0, [tmpt, tmpth, tmpEMD], axis=0)

            # Plotting Optimization Matrix
            ax1.pcolormesh(xx, yy, ssn_data.EMDD[siInx], alpha=1, linewidth=2, vmin=np.min(ssn_data.EMDD[siInx]),
                           vmax=6 * np.min(ssn_data.EMDD[siInx]))

            # True Interval
            ax1.scatter(OpMat[0, 0], OpMat[0, 1], c='r', edgecolors='w', linewidths=2, s=200, zorder=11)

            # Best 5 points
            for i in range(1, 5):
                ax1.scatter(OpMat[i, 0], OpMat[i, 1], c='w', linewidths=2, s=150, zorder=11, alpha=0.5)
                ax2.plot(
                    ssn_data.obsPlt['X'][
                        np.logical_and(ssn_data.obsPlt['X'] >= np.min(TObsFYr),
                                       ssn_data.obsPlt['X'] < np.max(TObsFYr))] -
                    ssn_data.cenPoints['OBS'][siInx, 0] +
                    OpMat[i, 0]
                    , ssn_data.obsPlt['Y'][
                        np.logical_and(ssn_data.obsPlt['X'] >= np.min(TObsFYr),
                                       ssn_data.obsPlt['X'] < np.max(TObsFYr))],
                    color=ssn_data.Clr[5 % i], linewidth=3
                    , alpha=0.2)

            # Best 5-10 points
            for i in range(5, 10):
                ax1.scatter(OpMat[i, 0], OpMat[i, 1], c='w', linewidths=2, s=100, zorder=11, alpha=0.5)
                ax2.plot(
                    ssn_data.obsPlt['X'][
                        np.logical_and(ssn_data.obsPlt['X'] >= np.min(TObsFYr),
                                       ssn_data.obsPlt['X'] < np.max(TObsFYr))] -
                    ssn_data.cenPoints['OBS'][siInx, 0] +
                    OpMat[i, 0]
                    , ssn_data.obsPlt['Y'][
                        np.logical_and(ssn_data.obsPlt['X'] >= np.min(TObsFYr),
                                       ssn_data.obsPlt['X'] < np.max(TObsFYr))],
                    color=ssn_data.Clr[5 % i], linewidth=3
                    , alpha=0.2)

            # Best 10-15 points
            for i in range(10, 15):
                ax1.scatter(OpMat[i, 0], OpMat[i, 1], c='w', linewidths=2, s=50, zorder=11, alpha=0.5)

            # Best 15-20 points
            for i in range(15, 100):
                ax1.scatter(OpMat[i, 0], OpMat[i, 1], c='w', linewidths=2, s=1, zorder=11, alpha=0.5)

            # Masking Gaps
            pltMsk = np.logical_not(cadMask)
            ax1.fill_between(ssn_data.REF_Dat['FRACYEAR'], ssn_data.REF_Dat['FRACYEAR'] * 0,
                             y2=ssn_data.REF_Dat['FRACYEAR'] * 0 + ssn_data.thN,
                             where=pltMsk, color='w', zorder=10)

            # Plotting real location
            ax1.plot(np.array([1, 1]) * TObsFYr[obsMinInx], np.array([0, np.max(y)]), 'w--', linewidth=3)

        # Plotting edges
        ax1.plot(np.array([1, 1]) * np.min(TObsFYr), np.array([0, np.max(y)]), ':', zorder=11, linewidth=3,
                 color=ssn_data.Clr[siInx % 6])
        ax1.plot(np.array([1, 1]) * np.max(TObsFYr), np.array([0, np.max(y)]), ':', zorder=11, linewidth=3,
                 color=ssn_data.Clr[siInx % 6])

        ax2.plot(np.array([1, 1]) * np.min(TObsFYr), np.array([0, np.max(y)]), ':', zorder=11, linewidth=3,
                 color=ssn_data.Clr[siInx % 6])
        ax2.plot(np.array([1, 1]) * np.max(TObsFYr), np.array([0, np.max(y)]), '-', zorder=11, linewidth=1,
                 color=ssn_data.Clr[siInx % 6])
        ax2.plot(np.array([1, 1]) * np.max(TObsFYr), np.array([0, np.max(y)]), ':', zorder=11, linewidth=3,
                 color=ssn_data.Clr[siInx % 6])

        # Axes properties
        ax1.set_ylabel('Area threshold (uHem)')
        ax1.set_xlim(left=np.min(ssn_data.REF_Dat['FRACYEAR']), right=np.max(ssn_data.REF_Dat['FRACYEAR']))
        ax1.set_ylim(bottom=0, top=np.max(y))

        ax1.spines['bottom'].set_color(ssn_data.Clr[siInx % 6])
        ax1.spines['bottom'].set_linewidth(3)
        ax1.spines['top'].set_color(ssn_data.Clr[siInx % 6])
        ax1.spines['top'].set_linewidth(3)
        ax1.spines['right'].set_color(ssn_data.Clr[siInx % 6])
        ax1.spines['right'].set_linewidth(3)
        ax1.spines['left'].set_color(ssn_data.Clr[siInx % 6])
        ax1.spines['left'].set_linewidth(3)

        # Adding title
        if siInx == ssn_data.cenPoints['OBS'].shape[0] - 1:
            # ax1.text(0.5, 1.01,'Chi-Square (y-y_exp)^2/(y^2+y_exp^2) for ' + NamObs.capitalize(), horizontalalignment='center', transform = ax1.transAxes)
            ax1.text(0.5, 1.01, 'EMD linear distance for ' + ssn_data.NamObs, horizontalalignment='center',
                     transform=ax1.transAxes)

    fig.savefig(figure_path, bbox_inches='tight')

    print('done.', flush=True)
    print(' ', flush=True)


def plotDistributionOfThresholdsMI(ssn_data,
                                   dpi=300,
                                   pxx=1500,
                                   pxy=1500,
                                   padv=50,
                                   padh=50,
                                   padv2=100,
                                   padh2=100):
    """

    :param dpi: Dots per inch in figure
    :param pxx: Horizontal size of each panel in pixels
    :param pxy: Vertical size of each panel in pixels
    :param padv: Vertical padding in pixels at the edge of the figure in pixels
    :param padh: Horizontal padding in pixels at the edge of the figure in pixels
    :param padv2: Vertical padding in pixels between panels
    :param padh2: Horizontal padding in pixels between panels
    """

    font = ssn_data.font
    plt.rc('font', **font)

    print('Creating and saving distribution of thresholds for different intervals figure...', end="", flush=True)

    figure_path = config.get_file_output_string('04', 'Distribution_of_Thresholds_MI',
                                                ssn_data=ssn_data,
                                                num_type=config.NUM_TYPE,
                                                den_type=config.DEN_TYPE)

    if config.SKIP_PRESENT_PLOTS and os.path.exists(figure_path):
        print(
            "\nFigure at {} already exists.\n Change the OVERWRITE_OBSERVERS config flag to overwrite existing plots\n".format(
                figure_path).format(figure_path))
        return

    frc = 0.8  # Fraction of the panel devoted to histograms

    nph = 3  # Number of horizontal panels
    npv = int(np.ceil(ssn_data.vldIntr.shape[0] / nph))  # Number of vertical panels

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
            if n < ssn_data.vldIntr.shape[0]:

                # Plot only if the period is valid
                if ssn_data.vldIntr[n]:
                    # Top Distribution
                    axd = fig.add_axes(
                        [ppadh + i * (pxx / fszh + ppadh2), ppadv + j * (pxy / fszv + ppadv2) + pxy / fszv * frc,
                         pxx / fszh * frc, pxy / fszv * (1 - frc)], label='a' + str(n))
                    axd.hist(ssn_data.bestTh[n][:, 2], bins=(np.arange(0, config.NBEST, 2)) / config.NBEST * (
                        np.ceil(np.max(ssn_data.bestTh[n][:, 2])) - np.floor(np.min(ssn_data.bestTh[n][:, 2])))
                                                            + np.floor(np.min(ssn_data.bestTh[n][:, 2])),
                             color=ssn_data.Clr[4],
                             alpha=.6,
                             density=True);

                    # Axes properties
                    axd.set_xlim(left=np.floor(np.min(ssn_data.bestTh[n][:, 2])),
                                 right=np.ceil(np.max(ssn_data.bestTh[n][:, 2])))
                    axd.set_axis_off()

                    # Right Distribution
                    ax2 = fig.add_axes(
                        [ppadh + i * (pxx / fszh + ppadh2) + pxx / fszh * frc, ppadv + j * (pxy / fszv + ppadv2),
                         pxx / fszh * frc * (1 - frc), pxy / fszv * frc], label='b' + str(n))
                    ax2.hist(ssn_data.bestTh[n][:, 1], bins=np.arange(0, ssn_data.thN, ssn_data.thI * 2),
                             color=ssn_data.Clr[2],
                             alpha=.6,
                             orientation='horizontal', density=True);

                    # # Axes properties
                    ax2.set_ylim(bottom=0, top=ssn_data.thN * ssn_data.thI)
                    ax2.set_axis_off()

                    # Scatter Plot
                    ax1 = fig.add_axes(
                        [ppadh + i * (pxx / fszh + ppadh2), ppadv + j * (pxy / fszv + ppadv2), pxx / fszh * frc,
                         pxy / fszv * frc], sharex=axd, label='b' + str(n))
                    ax1.scatter(ssn_data.bestTh[n][:, 2], ssn_data.bestTh[n][:, 1], color="0.25", edgecolor="k",
                                alpha=0.1,
                                s=100,
                                linewidths=2)

                    ax1.plot(np.array(
                        [np.floor(np.min(ssn_data.bestTh[n][:, 2])), np.ceil(np.max(ssn_data.bestTh[n][:, 2]))]),
                        np.array([1, 1]) * ssn_data.wAvI[n], '--'
                        , color=ssn_data.Clr[4], linewidth=3)
                    ax1.plot(np.array(
                        [np.floor(np.min(ssn_data.bestTh[n][:, 2])), np.ceil(np.max(ssn_data.bestTh[n][:, 2]))]),
                        np.array([1, 1]) * ssn_data.wAvI[n] - ssn_data.wSDI[n], ':'
                        , color=ssn_data.Clr[4], linewidth=2)
                    ax1.plot(np.array(
                        [np.floor(np.min(ssn_data.bestTh[n][:, 2])), np.ceil(np.max(ssn_data.bestTh[n][:, 2]))]),
                        np.array([1, 1]) * ssn_data.wAvI[n] + ssn_data.wSDI[n], ':'
                        , color=ssn_data.Clr[4], linewidth=2)

                    # Axes properties
                    ax1.set_ylabel('Area threshold (uHem)')
                    ax1.set_xlabel('EMD for ' + ssn_data.NamObs)
                    ax1.text(0.5, 0.95,
                             'From ' + str(np.round(ssn_data.endPoints['OBS'][n, 0], decimals=2)) + '  to ' + str(
                                 np.round(ssn_data.endPoints['OBS'][n + 1, 0], decimals=2)),
                             horizontalalignment='center',
                             verticalalignment='center', transform=ax1.transAxes)
                    ax1.set_xlim(left=np.floor(np.min(ssn_data.bestTh[n][:, 2])),
                                 right=np.ceil(np.max(ssn_data.bestTh[n][:, 2])))
                    ax1.set_ylim(bottom=0, top=ssn_data.thN * ssn_data.thI);

    fig.savefig(figure_path, bbox_inches='tight')

    print('done.', flush=True)
    print(' ', flush=True)


def _plotHistSqrtSSN(ssn_data, ax, calRefT, calObsT, Th):

    """

    :param ax: Axis handle to plot the 2D histogram
    :param calRefT: Group counts for reference observed after application of threshold
    :param calObsT: Group counts of calibrated observer
    :param Th: Optimal threshold used to obtain data in this plot.  It is printed at the top of the figure
    :param maxN: Maximum number of groups observed by the reference or calibrated observers in period of interest.  Used to set axis limits.
    :param SqrtF: Flag that determines whether the 2D histogram and metrics are calculated with the sqrt(GN+1) 'true' or GN 'false'
    """

    # Number of bins to use
    maxN = ssn_data.maxNPlt
    Nbins = ssn_data.maxNPlt

    # Applying Sqrt + 1
    if config.SQRT_2DHIS:
        maxN = np.sqrt(maxN)
        calRefT = np.sqrt(calRefT + 1)
        calObsT = np.sqrt(calObsT + 1)

    # Average group number
    ax.hist2d(calObsT, calRefT, bins=ssn_data.edges, cmap=plt.cm.magma_r, cmin=1)

    # Calculating Quantities for plot
    hist, xedges, yedges = np.histogram2d(calObsT, calRefT, bins=ssn_data.edges)
    alphaY = np.sum(hist, axis=1) / np.max(np.sum(hist, axis=1)) * 0.95 + 0.05

    # Calculating Quantities for plot and plot centers in Y
    Ymedian = ssn_data.centers * np.nan
    for i in range(0, ssn_data.centers.shape[0]):
        ypoints = calRefT[np.logical_and(calObsT >= ssn_data.edges[i], calObsT <= ssn_data.edges[i + 1])]
        if ypoints.shape[0] > 0:
            Ymedian[i] = np.nanmedian(ypoints)
            pecentilesy = np.abs(np.percentile(ypoints, np.array([15, 85]), interpolation='linear') - Ymedian[i])

            xpoints = calObsT[np.logical_and(calRefT >= (Ymedian[i] - (np.ceil(maxN)) / Nbins / 2),
                                             calRefT <= (Ymedian[i] + (np.ceil(maxN)) / Nbins / 2))]

            if xpoints.shape[0] > 0:
                pecentilesx = np.abs(np.percentile(xpoints, np.array([15, 85]), interpolation='linear') - ssn_data.centers[i])
            else:
                pecentilesx = pecentilesy * 0

            ax.errorbar(ssn_data.centers[i], Ymedian[i], yerr=np.expand_dims(pecentilesy, axis=1),
                        xerr=np.expand_dims(pecentilesx, axis=1), color='k', zorder=11, alpha=alphaY[i])
            ax.scatter(ssn_data.centers[i], Ymedian[i], color='w', edgecolor='k', s=100, linewidths=3, zorder=11,
                       alpha=alphaY[i])

    # Calculating quantities for assessment
    y = Ymedian
    x = ssn_data.centers

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

    ax.plot(ssn_data.edges, ssn_data.edges, '--'
            , color=ssn_data.Clr[4], linewidth=3)

    ax.text(0.5, 0.95,
            '$R^2M$=' + str(np.round(rSq, decimals=2)) + ' $MR$=' + str(np.round(mRes, decimals=2)) + ' $MRR$=' + str(int(np.round(mRRes*100,decimals=0))) + '% $Th$=' + str(
                Th), horizontalalignment='center', verticalalignment='center', transform=ax.transAxes)

    # Axes properties

    if config.SQRT_2DHIS:
        ax.set_xlabel('sqrt(GN+1) for ' + ssn_data.NamObs.capitalize())
        ax.set_ylabel('sqrt(GN+1) for reference')
    else:
        ax.set_xlabel('GN for ' + ssn_data.NamObs.capitalize())
        ax.set_ylabel('GN for reference observer')

    ax.set_xticks(np.arange(ssn_data.centers[0], np.ceil(ssn_data.centers[-1]), np.floor((np.ceil(ssn_data.centers[-1]) - ssn_data.centers[0]) / 8) + 1))
    ax.set_yticks(np.arange(ssn_data.centers[0], np.ceil(ssn_data.centers[-1]), np.floor((np.ceil(ssn_data.centers[-1]) - ssn_data.centers[0]) / 8) + 1))


def plotIntervalScatterPlots(ssn_data,
                             dpi=300,
                             pxx=2300,
                             pxy=2300,
                             padv=50,
                             padh=50,
                             padv2=100,
                             padh2=100):
    """

    :param dpi: Dots per inch in figure
    :param pxx: Horizontal size of each panel in pixels
    :param pxy: Vertical size of each panel in pixels
    :param padv: Vertical padding in pixels at the edge of the figure in pixels
    :param padh: Horizontal padding in pixels at the edge of the figure in pixels
    :param padv2: Vertical padding in pixels between panels
    :param padh2: Horizontal padding in pixels between panels
    """

    font = ssn_data.font
    plt.rc('font', **font)

    print('Creating and saving interval scatter-plots figure...', end="", flush=True)

    figure_path = config.get_file_output_string('05', 'Interval_Scatter_Plots',
                                                ssn_data=ssn_data,
                                                num_type=config.NUM_TYPE,
                                                den_type=config.DEN_TYPE)

    if config.SKIP_PRESENT_PLOTS and os.path.exists(figure_path):
        print("\nFigure at {} already exists.\n"
              " Change the OVERWRITE_OBSERVERS config flag to overwrite existing plots\n".format(
            figure_path))
        return

    frc = 0.8  # Fraction of the panel devoted to histograms

    nph = 3  # Number of horizontal panels
    npv = int(np.ceil(ssn_data.vldIntr.shape[0] / nph))  # Number of vertical panels

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
            if n < ssn_data.vldIntr.shape[0]:

                # Plot only if the period is valid and has overlap
                if ssn_data.vldIntr[n] and np.sum(
                        np.logical_and(ssn_data.REF_Dat['FRACYEAR'] >= ssn_data.endPoints['OBS'][n, 0],
                                       ssn_data.REF_Dat['FRACYEAR'] < ssn_data.endPoints['OBS'][
                                                   n + 1, 0])) > 0:
                    grpsREFw = ssn_data.calRef[n]
                    grpsObsw = ssn_data.calObs[n]

                    # Average group number
                    ax1 = fig.add_axes(
                        [ppadh + i * (pxx / fszh + ppadh2), ppadv + j * (pxy / fszv + ppadv2), pxx / fszh * frc,
                         pxy / fszv * frc], label='b' + str(n))

                    _plotHistSqrtSSN(ssn_data, ax1, grpsREFw, grpsObsw, np.round(ssn_data.wAvI[n], decimals=1))

                    ax1.text(0.5, 0.87,
                             'From ' + str(np.round(ssn_data.endPoints['OBS'][n, 0], decimals=2)) + '  to ' + str(
                                 np.round(ssn_data.endPoints['OBS'][n + 1, 0], decimals=2)),
                             horizontalalignment='center',
                             verticalalignment='center', transform=ax1.transAxes)



    fig.savefig(figure_path, bbox_inches='tight')

    print('done.', flush=True)
    print(' ', flush=True)


def plotMinEMD(ssn_data,
               dpi=300,
               pxx=4000,
               pxy=1000,
               padv=30,
               padh=30,
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

    print('Creating and saving minimum EMD figure...', end="", flush=True)

    figure_path = config.get_file_output_string('06', 'Min_EMD',
                                                ssn_data=ssn_data,
                                                num_type=config.NUM_TYPE,
                                                den_type=config.DEN_TYPE)

    if config.SKIP_PRESENT_PLOTS and os.path.exists(figure_path):
        print(
            "\nFigure at {} already exists.\n Change the OVERWRITE_OBSERVERS config flag to overwrite existing plots\n".format(
                figure_path))
        return

    font = ssn_data.font
    plt.rc('font', **font)

    nph = 1  # Number of horizontal panels
    npv = ssn_data.cenPoints['OBS'].shape[0] + 1  # Number of vertical panels

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
    ax2.plot(ssn_data.REF_Dat['FRACYEAR'], ssn_data.REF_Dat['AVGROUPS'], 'r--', linewidth=2, alpha=1)

    # Plotting Observer
    ax2.plot(ssn_data.obsPlt['X'], ssn_data.obsPlt['Y'], color=ssn_data.Clr[0], linewidth=2)

    # Axes properties
    ax2.set_ylabel('Average Number of Groups')
    ax2.set_xlabel('Center of sliding window (Year)')
    ax2.set_xlim(left=np.min(ssn_data.REF_Dat['FRACYEAR']), right=np.max(ssn_data.REF_Dat['FRACYEAR']));
    ax2.set_ylim(bottom=0, top=np.max(ssn_data.REF_Dat['AVGROUPS']) * 1.1)

    # Going through different sub-intervals
    for siInx in range(0, ssn_data.cenPoints['OBS'].shape[0]):

        # Defining mask based on the interval type (rise or decay)
        if ssn_data.cenPoints['OBS'][siInx, 1] > 0:
            cadMaskI = ssn_data.risMask['INDEX']
            cadMask = ssn_data.risMask['PLOT']
        else:
            cadMaskI = ssn_data.decMask['INDEX']
            cadMask = ssn_data.decMask['PLOT']

        # Creating axis
        ax1 = fig.add_axes([ppadh, ppadv + (siInx + 1) * (pxy / fszv + ppadv2), pxx / fszh, pxy / fszv])

        TObsFYr = ssn_data.ObsDat.loc[
            np.logical_and(ssn_data.ObsDat['FRACYEAR'] >= ssn_data.endPoints['OBS'][siInx, 0],
                           ssn_data.ObsDat['FRACYEAR'] < ssn_data.endPoints['OBS'][siInx + 1, 0])
            , 'FRACYEAR'].values.copy()

        # Initializing varialble to plot non-valid intervals
        x = ssn_data.REF_Dat['FRACYEAR'].values[cadMaskI]
        y = np.array([1, 10])

        # Plot only if period is valid
        if ssn_data.vldIntr[siInx]:
            # Calculating minimum distance for plotting
            y = np.amin(ssn_data.EMDD[siInx], axis=0)

            # Plotting Optimization Matrix
            ax1.plot(x, y, color='k', linewidth=3)

            # Masking Gaps
            pltMsk = np.logical_not(cadMask)
            ax1.fill_between(ssn_data.REF_Dat['FRACYEAR'], ssn_data.REF_Dat['FRACYEAR'] * 0,
                             y2=ssn_data.REF_Dat['FRACYEAR'] * 0 + np.min(y) * 10, where=pltMsk, color='w', zorder=10)

            # Plotting possible theshold
            ax1.plot(np.array([np.min(x), np.max(x)]), np.array([1, 1]) * ssn_data.disThres * np.min(y), 'b:',
                     linewidth=3)

        # Plotting edges
        ax1.plot(np.array([1, 1]) * np.min(TObsFYr), np.array([0, np.max(y)]), ':', zorder=11, linewidth=3,
                 color=ssn_data.Clr[siInx % 6])
        ax1.plot(np.array([1, 1]) * np.max(TObsFYr), np.array([0, np.max(y)]), ':', zorder=11, linewidth=3,
                 color=ssn_data.Clr[siInx % 6])

        ax2.plot(np.array([1, 1]) * np.min(TObsFYr), np.array([0, np.max(y)]), ':', zorder=11, linewidth=3,
                 color=ssn_data.Clr[siInx % 6])
        ax2.plot(np.array([1, 1]) * np.max(TObsFYr), np.array([0, np.max(y)]), '-', zorder=11, linewidth=1,
                 color=ssn_data.Clr[siInx % 6])
        ax2.plot(np.array([1, 1]) * np.max(TObsFYr), np.array([0, np.max(y)]), ':', zorder=11, linewidth=3,
                 color=ssn_data.Clr[siInx % 6])

        # Axes properties
        ax1.set_ylabel('Distribution Distance')
        ax1.set_xlim(left=np.min(ssn_data.REF_Dat['FRACYEAR']), right=np.max(ssn_data.REF_Dat['FRACYEAR']))
        ax1.set_ylim(bottom=0, top=np.min(y) * 5 + 1)

        ax1.spines['bottom'].set_color(ssn_data.Clr[siInx % 6])
        ax1.spines['bottom'].set_linewidth(3)
        ax1.spines['top'].set_color(ssn_data.Clr[siInx % 6])
        ax1.spines['top'].set_linewidth(3)
        ax1.spines['right'].set_color(ssn_data.Clr[siInx % 6])
        ax1.spines['right'].set_linewidth(3)
        ax1.spines['left'].set_color(ssn_data.Clr[siInx % 6])
        ax1.spines['left'].set_linewidth(3)

        # Adding title
        if siInx == ssn_data.cenPoints['OBS'].shape[0] - 1:
            # ax1.text(0.5, 1.01,'Chi-Square (y-y_exp)^2/(y^2+y_exp^2) for ' + NamObs.capitalize(), horizontalalignment='center', transform = ax1.transAxes)
            ax1.text(0.5, 1.01, 'EMD linear distance for ' + ssn_data.NamObs, horizontalalignment='center',
                     transform=ax1.transAxes)

    fig.savefig(figure_path, bbox_inches='tight')

    print('done.', flush=True)
    print(' ', flush=True)



def plotSimultaneousFit(ssn_data,
                        dpi=300,
                        pxx=4000,
                        pxy=1000,
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

    print('Creating and saving simultaneous fit figure...', end="", flush=True)

    figure_path = config.get_file_output_string('07', 'Simultaneous_Fit',
                                                ssn_data=ssn_data,
                                                num_type=config.NUM_TYPE,
                                                den_type=config.DEN_TYPE)

    if config.SKIP_PRESENT_PLOTS and os.path.exists(figure_path):
        print(
            "\nFigure at {} already exists.\n Change the OVERWRITE_OBSERVERS config flag to overwrite existing plots\n".format(
                figure_path))
        return

    font = ssn_data.font
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
    ax2.plot(ssn_data.REF_Dat['FRACYEAR'], ssn_data.REF_Dat['AVGROUPS'], 'r--', linewidth=2, alpha=1)

    # Plotting Observer
    ax2.plot(ssn_data.obsPlt['X'], ssn_data.obsPlt['Y'], color=ssn_data.Clr[0], linewidth=2)

    # Axes properties
    ax2.set_ylabel('Av. Num. of Groups')
    ax2.set_xlabel('Center of sliding window (Year)')
    ax2.set_xlim(left=np.min(ssn_data.REF_Dat['FRACYEAR']), right=np.max(ssn_data.REF_Dat['FRACYEAR']));
    ax2.set_ylim(bottom=0, top=np.max(ssn_data.REF_Dat['AVGROUPS']) * 1.1)

    # Placement of top simultaneous fits
    ax1 = fig.add_axes([ppadh, ppadv + (pxy / fszv + ppadv2), pxx / fszh * frc, pxy / fszv])

    # Going through different sub-intervals
    for siInx in range(0, ssn_data.cenPoints['OBS'].shape[0]):
        TObsFYr = ssn_data.ObsDat.loc[
            np.logical_and(ssn_data.ObsDat['FRACYEAR'] >= ssn_data.endPoints['OBS'][siInx, 0],
                           ssn_data.ObsDat['FRACYEAR'] < ssn_data.endPoints['OBS'][siInx + 1, 0])
            , 'FRACYEAR'].values.copy()

        # Plotting edges
        ax1.plot(np.array([1, 1]) * np.min(TObsFYr), np.array([0, ssn_data.thN * ssn_data.thI]), ':', zorder=11,
                 linewidth=3,
                 color=ssn_data.Clr[siInx % 6])
        ax1.plot(np.array([1, 1]) * np.max(TObsFYr), np.array([0, ssn_data.thN * ssn_data.thI]), '-', zorder=11,
                 linewidth=1,
                 color=ssn_data.Clr[siInx % 6])
        ax1.plot(np.array([1, 1]) * np.max(TObsFYr), np.array([0, ssn_data.thN * ssn_data.thI]), ':', zorder=11,
                 linewidth=3,
                 color=ssn_data.Clr[siInx % 6])

        ax2.plot(np.array([1, 1]) * np.min(TObsFYr), np.array([0, ssn_data.thN * ssn_data.thI]), ':', zorder=11,
                 linewidth=3,
                 color=ssn_data.Clr[siInx % 6])
        ax2.plot(np.array([1, 1]) * np.max(TObsFYr), np.array([0, ssn_data.thN * ssn_data.thI]), '-', zorder=11,
                 linewidth=1,
                 color=ssn_data.Clr[siInx % 6])
        ax2.plot(np.array([1, 1]) * np.max(TObsFYr), np.array([0, ssn_data.thN * ssn_data.thI]), ':', zorder=11,
                 linewidth=3,
                 color=ssn_data.Clr[siInx % 6])

    for i in range(0, config.NBEST):

        # Initialize plot vector
        x = np.array([])

        for siInx in range(0, ssn_data.cenPoints['OBS'].shape[0]):

            # Defining mask based on the interval type (rise or decay)
            if ssn_data.cenPoints['OBS'][siInx, 1] > 0:
                cadMaskI = ssn_data.risMask['INDEX']
            else:
                cadMaskI = ssn_data.decMask['INDEX']

            Year = ssn_data.REF_Dat['FRACYEAR'].values[cadMaskI]

            # Append only if period is valid
            if ssn_data.vldIntr[siInx]:

                # If it is the first interval re-create the array
                if x.shape[0] == 0:
                    x = np.array([Year[ssn_data.EMDComb[siInx + 2, i].astype(np.int)]])
                # Append other sub-intervals
                else:
                    x = np.append(x, Year[ssn_data.EMDComb[siInx + 2, i].astype(np.int)])

                    # Creating matching threshold vector
        y = x * 0 + ssn_data.EMDComb[1, i]

        # Constructing alpha
        alph = 1 - (ssn_data.EMDComb[0, i] - np.min(ssn_data.EMDComb[0, :])) / (
            np.max(ssn_data.EMDComb[0, :]) - np.min(ssn_data.EMDComb[0, :]))

        # Plotting Intervals
        ax1.plot(x, y, 'o:', zorder=11, linewidth=1, color=ssn_data.Clr[2], alpha=alph, markersize=(101 - i) / 8)

    # Axes properties
    ax1.set_ylabel('Area thres. (uHem)')
    ax1.set_xlim(left=np.min(ssn_data.REF_Dat['FRACYEAR']), right=np.max(ssn_data.REF_Dat['FRACYEAR']))
    ax1.set_ylim(bottom=0, top=ssn_data.thN * ssn_data.thI)

    ax1.text(0.5, 1.01, 'Best Simultaneous Fits for ' + ssn_data.NamObs, horizontalalignment='center',
             transform=ax1.transAxes);

    # Right Distribution
    ax3 = fig.add_axes(
        [ppadh + pxx / fszh * frc, ppadv + (pxy / fszv + ppadv2), pxx / fszh * (1 - frc), pxy / fszv])
    ax3.hist(ssn_data.EMDComb[1, :], bins=np.arange(0, ssn_data.thN) * ssn_data.thI + ssn_data.thI / 2,
             color=ssn_data.Clr[2], alpha=.6,
             orientation='horizontal', density=True);
    # ax3.plot(yOD, xOD, color=Clr[2], linewidth=3)

    # # Axes properties
    ax3.set_ylim(bottom=0, top=ssn_data.thN * ssn_data.thI)
    ax3.set_axis_off()

    fig.savefig(figure_path, bbox_inches='tight')

    print('done.', flush=True)
    print(' ', flush=True)


def plotDistributionOfThresholds(ssn_data,
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

    font = ssn_data.font
    plt.rc('font', **font)

    print('Creating and saving distribution of thresholds for different intervals figure...', end="", flush=True)

    figure_path = config.get_file_output_string('08', 'Distribution_of_Thresholds',
                                                ssn_data=ssn_data,
                                                num_type=config.NUM_TYPE,
                                                den_type=config.DEN_TYPE)

    if config.SKIP_PRESENT_PLOTS and os.path.exists(figure_path):
        print(
            "\nFigure at {} already exists.\n Change the OVERWRITE_OBSERVERS config flag to overwrite existing plots\n".format(
                figure_path))
        return

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
    axd.hist(ssn_data.EMDComb[0, :],
             bins=(np.arange(0, config.NBEST, 2)) / config.NBEST * (
                 np.ceil(np.max(ssn_data.EMDComb[0, :])) - np.floor(np.min(ssn_data.EMDComb[0, :])))
                  + np.floor(np.min(ssn_data.EMDComb[0, :])), color=ssn_data.Clr[4], alpha=.6, density=True)

    # Axes properties
    axd.set_xlim(left=np.floor(np.min(ssn_data.EMDComb[0, :])), right=np.ceil(np.max(ssn_data.EMDComb[0, :])))
    axd.set_axis_off()

    # Right Distribution
    ax2 = fig.add_axes([ppadh + pxx / fszh * frc, ppadv, pxx / fszh * frc * (1 - frc), pxy / fszv * frc])
    ax2.hist(ssn_data.EMDComb[1, :], bins=np.arange(0, ssn_data.thN, ssn_data.thI * 2), color=ssn_data.Clr[2], alpha=.6,
             orientation='horizontal', density=True);
    # ax2.plot(yOD, xOD, color=Clr[2], linewidth=3)

    # # Axes properties
    ax2.set_ylim(bottom=0, top=ssn_data.thN * ssn_data.thI)
    ax2.set_axis_off()

    # Scatter Plot
    ax1 = fig.add_axes([ppadh, ppadv, pxx / fszh * frc, pxy / fszv * frc], sharex=axd)
    ax1.scatter(ssn_data.EMDComb[0, :], ssn_data.EMDComb[1, :], color="0.25", edgecolor="k", alpha=0.1, s=100,
                linewidths=2)

    ax1.plot(np.array([np.floor(np.min(ssn_data.EMDComb[0, :])), np.ceil(np.max(ssn_data.EMDComb[0, :]))]),
             np.array([1, 1]) * ssn_data.wAv,
             '--'
             , color=ssn_data.Clr[4], linewidth=3)
    ax1.plot(np.array([np.floor(np.min(ssn_data.EMDComb[0, :])), np.ceil(np.max(ssn_data.EMDComb[0, :]))]),
             np.array([1, 1]) * ssn_data.wAv - ssn_data.wSD, ':'
             , color=ssn_data.Clr[4], linewidth=2)
    ax1.plot(np.array([np.floor(np.min(ssn_data.EMDComb[0, :])), np.ceil(np.max(ssn_data.EMDComb[0, :]))]),
             np.array([1, 1]) * ssn_data.wAv + ssn_data.wSD, ':'
             , color=ssn_data.Clr[4], linewidth=2)

    # Axes properties
    ax1.set_ylabel('Area threshold (uHem)')
    ax1.text(1.02, 0.52, 'Area threshold (uHem)', horizontalalignment='center', transform=ax1.transAxes,
             rotation='vertical', verticalalignment='center')
    ax1.set_xlabel('EMD')
    ax1.text(0.5, 1.01, 'EMD for ' + ssn_data.NamObs, horizontalalignment='center', transform=ax1.transAxes)
    ax1.set_xlim(left=np.floor(np.min(ssn_data.EMDComb[0, :])), right=np.ceil(np.max(ssn_data.EMDComb[0, :])))
    ax1.set_ylim(bottom=0, top=ssn_data.thN * ssn_data.thI)

    fig.savefig(figure_path, bbox_inches='tight')

    print('done.', flush=True)
    print(' ', flush=True)


def plotSingleThresholdScatterPlot(ssn_data,
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

    font = ssn_data.font
    plt.rc('font', **font)

    print('Creating and saving scatterplot of overlap...', end="",
          flush=True)

    figure_path = config.get_file_output_string('09', 'Single_Threshold_ScatterPlot',
                                                ssn_data=ssn_data,
                                                num_type=config.NUM_TYPE,
                                                den_type=config.DEN_TYPE)

    if config.SKIP_PRESENT_PLOTS and os.path.exists(figure_path):
        print(
            "\nFigure at {} already exists.\n Change the OVERWRITE_OBSERVERS config flag to overwrite existing plots\n".format(
                figure_path))
        return

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

    # Determine which threshold to use
    if np.sum(ssn_data.vldIntr) > 1:
        Th = ssn_data.wAv
    else:
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

    # Average group number
    ax1 = fig.add_axes([ppadh, ppadv, pxx / fszh, pxy / fszv])

    _plotHistSqrtSSN(ssn_data, ax1, grpsREFw, grpsObsw, np.round(Th, decimals=1))

    ax1.set_title('Single threshold - All days of overlap')

    fig.savefig(figure_path, bbox_inches='tight')

    print('done.', flush=True)
    print(' ', flush=True)


def plotMultiThresholdScatterPlot(ssn_data,
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

    font = ssn_data.font
    plt.rc('font', **font)

    tcalRef = np.concatenate(ssn_data.calRef, axis=0)
    tcalObs = np.concatenate(ssn_data.calObs, axis=0)

    # Only if there is at leas one interval that is valid
    if tcalRef.shape[0] > 1:

        print('Creating and saving scatterplot of overlap with different thresholds...', end="",
              flush=True)

        figure_path = config.get_file_output_string('10', 'Multi_Threshold_ScatterPlot',
                                                    ssn_data=ssn_data,
                                                    num_type=config.NUM_TYPE,
                                                    den_type=config.DEN_TYPE)

        if config.SKIP_PRESENT_PLOTS and os.path.exists(figure_path):
            print(
                "\nFigure at {} already exists.\n"
                " Change the OVERWRITE_OBSERVERS config flag to overwrite existing plots\n".format(figure_path))
            return

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

        for n in range(0, ssn_data.cenPoints['OBS'].shape[0]):

            # Plot only if the period is valid and has overlap
            if ssn_data.vldIntr[n] and np.sum(
                    np.logical_and(ssn_data.REF_Dat['FRACYEAR'] >= ssn_data.endPoints['OBS'][n, 0],
                                   ssn_data.REF_Dat['FRACYEAR'] < ssn_data.endPoints['OBS'][
                                               n + 1, 0])) > 0:
                # Calculating number of groups in reference data for given threshold
                grpsREFw = np.nansum(
                    np.greater(ssn_data.REF_Dat.values[:, 3:ssn_data.REF_Dat.values.shape[1] - 3], ssn_data.wAv),
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

        ax1 = fig.add_axes([ppadh, ppadv, pxx / fszh, pxy / fszv], label='b1')

        _plotHistSqrtSSN(ssn_data, ax1, calObsN, calRefN, np.round(ssn_data.wAv, decimals=1))

        # Average group number
        ax2 = fig.add_axes([ppadh + (pxx / fszh + ppadh2), ppadv, pxx / fszh, pxy / fszv], label='b2')

        _plotHistSqrtSSN(ssn_data, ax2, tcalObs, tcalRef, 'Variable')

        fig.savefig(figure_path, bbox_inches='tight')

        print('done.', flush=True)
        print(' ', flush=True)
