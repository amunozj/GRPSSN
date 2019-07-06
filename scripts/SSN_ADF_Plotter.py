import numpy as np
from astropy import convolution as conv
import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt
from matplotlib import colors as clrs
from pyemd import emd
from SSN_Config import SSN_ADF_Config as config
import os


def plotSearchWindows(ssn_data, SILSO_Sn, SIL_max, SIL_min, REF_min, REF_max,
                      dpi=325,
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
    ppadh = padh / fszv  # Horizontal padding the edge of the figure in relative units

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

    fig.clear()
    plt.close(fig)

    print('done.', flush=True)
    print(' ', flush=True)


def plotHistSnADF(ssn_data,
                  dpi=325,
                  pxx=1500,
                  pxy=1500,
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

    prepend = "PL" + str(config.PCTLO) + "_PH" + str(config.PCTHI)
    prepend += "QD" + str(config.QTADF) + "_AD" + str(config.ACADF)

    figure_path = '{}/02_'.format(ssn_data.output_path) + prepend + '_2DHist_SN_vs_ADF.png'

    if config.SKIP_PRESENT_PLOTS and os.path.exists(figure_path):
        print(
            "\nFigure at {} already exists.\n Change the OVERWRITE_OBSERVERS config flag to overwrite existing plots\n".format(
                figure_path).format(figure_path))
        return

    print('Creating and saving SN vs ADF Histogram figure...', end="", flush=True)

    # creating matrix to define thresholds
    TREFDat = ssn_data.REF_Grp['GROUPS'].values.copy()
    TREFSNd = ssn_data.REF_Grp['AVGSNd'].values.copy()

    GDREF = np.zeros((ssn_data.thNPc, np.int(TREFDat.shape[0] / ssn_data.MoLngt)))
    ODREF = np.zeros((ssn_data.thNPc, np.int(TREFDat.shape[0] / ssn_data.MoLngt)))
    SNdREF = np.zeros((ssn_data.thNPc, np.int(TREFDat.shape[0] / ssn_data.MoLngt)))

    for TIdx in range(0, ssn_data.thNPc):
        grpsREFw = np.nansum(
            np.greater(ssn_data.REF_Dat.values[:, 3:ssn_data.REF_Dat.values.shape[1] - 3], TIdx * ssn_data.thIPc),
            axis=1).astype(float)
        grpsREFw[np.isnan(ssn_data.REF_Dat['AREA1'])] = np.nan

        TgrpsREF = grpsREFw[0:np.int(grpsREFw.shape[0] / ssn_data.MoLngt) * ssn_data.MoLngt].copy()
        TgrpsREF = TgrpsREF.reshape((-1, ssn_data.MoLngt))
        TSNdREF = TREFSNd[0:np.int(TREFSNd.shape[0] / ssn_data.MoLngt) * ssn_data.MoLngt].copy()
        TSNdREF = TSNdREF.reshape((-1, ssn_data.MoLngt))
        # Number of days with groups
        GDREF[TIdx, :] = np.sum(np.greater(TgrpsREF, 0), axis=1)
        # Number of days with observations
        ODREF[TIdx, :] = np.sum(np.isfinite(TgrpsREF), axis=1)
        # ACTIVE DAY FRACTION
        ADFREF = GDREF / ODREF
        # Monthly sunspot number
        SNdREF[TIdx, :] = np.mean(TSNdREF, axis=1)

        # Plotting threshold
    plt.rc('font', **font)

    nph = 2  # Number of horizontal panels
    npv = ssn_data.thNPc  # Number of vertical panels

    # Figure sizes in pixels
    fszv = (npv * pxy + 2 * padv + (npv - 1) * padv2)  # Vertical size of figure in inches
    fszh = (nph * pxx + 2 * padh + (nph - 1) * padh2)  # Horizontal size of figure in inches

    # Conversion to relative unites
    ppadv = padv / fszv  # Vertical padding in relative units
    ppadh = padh / fszv  # Horizontal padding the edge of the figure in relative units

    Nbinsx = 20
    Nbinsy = ssn_data.MoLngt + 1

    edgesx = np.arange(0, Nbinsx + 1) / Nbinsx * 150
    edgesy = np.arange(0, Nbinsy + 1) / Nbinsy

    # Box plot AL range for percentile scan
    bprange = np.arange(10, 175, 10)

    # Curve AL range for percentile scan
    pprange = np.arange(5, 175, 2)

    # Start Figure
    fig = plt.figure(figsize=(fszh / dpi, fszv / dpi))

    LowALlim = np.zeros(ssn_data.thNPc)
    HighALlim = np.zeros(ssn_data.thNPc)

    for n in range(0, ssn_data.thNPc):

        pltmsk = ODREF[n, :] == ssn_data.MoLngt

        # ax1
        ax1 = fig.add_axes([ppadh, ppadv + n * pxy / fszv, pxx / fszh, pxy / fszv], label='b1')
        ax1.hist2d(SNdREF[n, :][pltmsk], ADFREF[n, :][pltmsk], bins=[edgesx, edgesy], cmap=plt.cm.magma_r, cmin=1)

        bpdat = []
        for AL in bprange:
            bpdat.append(ADFREF[n, :][np.logical_and(pltmsk, SNdREF[n, :] <= AL)])

        ax1.boxplot(bpdat, positions=bprange, widths=5)

        # Activity level percentile
        ADFP = pprange * np.nan
        # ADF below which we have config.PCTLO % of all ADFS for a given AL
        # ADF below which we have config.PCTLO % of all ADFS for a given AL
        for ALi in np.arange(0, pprange.shape[0]):
            if np.sum(np.logical_and(pltmsk, SNdREF[n, :] <= pprange[ALi])) > 0:
                ADFP[ALi] = np.percentile(ADFREF[n, :][np.logical_and(pltmsk, SNdREF[n, :] <= pprange[ALi])],
                                          config.PCTLO)

        ax1.plot(pprange, ADFP)
        ax1.plot([0, 150], [config.QTADF, config.QTADF], color='k', linestyle='--')

        # Intersect between our definition of what a quiet interval is and ADFP
        intrsc = np.where(np.abs(ADFP - config.QTADF) == np.nanmin(np.abs(ADFP - config.QTADF)))[0]
        cut = np.mean(pprange[intrsc])
        if np.sum(ADFP < config.QTADF) == 0:
            cut = np.nan

        LowALlim[n] = cut

        ax1.plot([cut, cut], [0, 1.2], color='k', linestyle=':')

        # Axes properties
        ax1.text(0.5, 0.9, 'Th: ' + str(n * ssn_data.thIPc) + ' - AL cut: ' + str(cut), horizontalalignment='center',
                 fontsize=15,
                 transform=ax1.transAxes)
        ax1.set_ylabel('ADF')
        ax1.set_ylim(top=1.2, bottom=0)
        ax1.set_xlim(left=0, right=150)

        # ax2
        ax2 = fig.add_axes([ppadh + pxx / fszh, ppadv + n * pxy / fszv, pxx / fszh, pxy / fszv], label='b2')
        ax2.hist2d(SNdREF[n, :][pltmsk], ADFREF[n, :][pltmsk], bins=[edgesx, edgesy], cmap=plt.cm.magma_r, cmin=1)

        bpdat = []
        for AL in bprange:
            bpdat.append(ADFREF[n, :][np.logical_and(pltmsk, SNdREF[n, :] >= AL)])

        ax2.boxplot(bpdat, positions=bprange, widths=5)

        # Activity level percentile
        ADFP = pprange * np.nan
        # ADF above which we have config.PCTHI % of all ADFS for a given AL
        for ALi in np.arange(0, pprange.shape[0]):
            if np.sum(np.logical_and(pltmsk, SNdREF[n, :] >= pprange[ALi])) > 0:
                ADFP[ALi] = np.percentile(ADFREF[n, :][np.logical_and(pltmsk, SNdREF[n, :] >= pprange[ALi])],
                                          100 - config.PCTHI)

        ax2.plot(pprange, ADFP)
        ax2.plot([0, 150], [config.ACADF, config.ACADF], color='k', linestyle='--')

        # Intersect between our definition of what an active interval is and ADFP
        intrsc = np.where(np.abs(ADFP - config.ACADF) == np.nanmin(np.abs(ADFP - config.ACADF)))[0]
        cut = np.mean(pprange[intrsc])
        if np.sum(ADFP < config.ACADF) == 0:
            cut = np.nan

        HighALlim[n] = cut

        ax2.plot([cut, cut], [0, 1.2], color='k', linestyle=':')

        # Axes properties
        ax2.set_ylabel('ADF')
        ax2.yaxis.set_label_position("right")
        ax2.text(0.5, 0.9, 'Th: ' + str(n * ssn_data.thIPc) + ' - AL cut: ' + str(cut), horizontalalignment='center',
                 fontsize=15,
                 transform=ax2.transAxes)
        ax2.set_ylim(top=1.2, bottom=0)

        ax2.yaxis.tick_right()
        ax2.set_xlim(left=0, right=150)

        if n > 0 & n < ssn_data.thNPc - 1:
            ax1.set_xticklabels([])
            ax2.set_xticklabels([])
        else:
            ax1.set_xlabel('SMSN')
            ax2.set_xlabel('SMSN')
            ax1.set_xticks([0, 50, 100, 150])
            ax1.set_xticklabels([0, 50, 100, 150])
            ax2.set_xticks([50, 100, 150])
            ax2.set_xticklabels([50, 100, 150])

        if n == ssn_data.thNPc - 1:
            ax1.xaxis.set_label_position("top")
            ax1.xaxis.tick_top()
            ax2.xaxis.tick_top()
            ax2.xaxis.set_label_position("top")
            ax1.set_xlabel('SMSN')
            ax2.set_xlabel('SMSN')
            ax1.set_xticks([0, 50, 100, 150])
            ax1.set_xticklabels([0, 50, 100, 150])
            ax2.set_xticks([50, 100, 150])
            ax2.set_xticklabels([50, 100, 150])

    fig.savefig(figure_path, bbox_inches='tight')

    fig.clear()
    plt.close(fig)

    print('done.', flush=True)
    print(' ', flush=True)


def plotFitAl(ssn_data,
              dpi=325,
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

    font = ssn_data.font
    plt.rc('font', **font)

    prepend = "PL" + str(config.PCTLO) + "_PH" + str(config.PCTHI)
    prepend += "QD" + str(config.QTADF) + "_AD" + str(config.ACADF)

    figure_path = '{}/03_'.format(ssn_data.output_path) + prepend + '_SN_vs_AL.png'

    if config.SKIP_PRESENT_PLOTS and os.path.exists(figure_path):
        print(
            "\nFigure at {} already exists.\n Change the OVERWRITE_OBSERVERS config flag to overwrite existing plots\n".format(
                figure_path).format(figure_path))
        return

    print('Creating and saving SN vs AL scatter plot figure...', end="", flush=True)

    # fit for low solar activity
    xlow = np.arange(0, ssn_data.thNPc) * ssn_data.thIPc
    xlow = xlow[np.isfinite(ssn_data.LowALlim)]
    ylow = ssn_data.LowALlim[np.isfinite(ssn_data.LowALlim)]

    # fit for high solar activity
    xhigh = np.arange(0, ssn_data.thNPc) * ssn_data.thIPc
    xhigh = xhigh[np.isfinite(ssn_data.HighALlim)]
    yhigh = ssn_data.HighALlim[np.isfinite(ssn_data.HighALlim)]

    plt.rc('font', **font)

    # Size definitions
    nph = 1  # Number of horizontal panels
    npv = 1  # Number of vertical panels

    # Padding

    # Figure sizes in pixels
    fszv = (npv * pxy + 2 * padv + (npv - 1) * padv2)  # Vertical size of figure in inches
    fszh = (nph * pxx + 2 * padh + (nph - 1) * padh2)  # Horizontal size of figure in inches

    # Conversion to relative unites
    ppadv = padv / fszv  # Vertical padding in relative units
    ppadh = padh / fszv  # Horizontal padding the edge of the figure in relative units

    # Start Figure
    fig = plt.figure(figsize=(fszh / dpi, fszv / dpi))
    ax1 = fig.add_axes([ppadh, ppadv, pxx / fszh, pxy / fszv])

    ax1.scatter(xlow, ylow, alpha=1)
    ax1.scatter(xhigh, yhigh, alpha=1)
    ax1.plot(xhigh, ssn_data.a1high * xhigh + ssn_data.a0high)
    ax1.plot(xlow, ssn_data.a1low * xlow + ssn_data.a0low)

    ax1.set_xlabel('SN Threshold (uHem)')
    ax1.set_ylabel('Activity Level Limit (SSN)')

    fig.savefig(figure_path, bbox_inches='tight')

    fig.clear()
    plt.close(fig)

    print('done.', flush=True)
    print(' ', flush=True)


def plotActiveVsObserved(ssn_data,
                         dpi=325,
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

    figure_path = config.get_file_output_string('04', 'active_vs_observed_days',
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

    yrOb = ssn_data.ObsDat['FRACYEAR'].values
    yrOb = yrOb[0:np.int(yrOb.shape[0] / ssn_data.MoLngt) * ssn_data.MoLngt]

    # Reshaping
    grpsOb = grpsOb.reshape((-1, ssn_data.MoLngt))
    yrOb = yrOb.reshape((-1, ssn_data.MoLngt))

    # Number of days with observations
    obsOb = np.sum(np.isfinite(grpsOb), axis=1)

    # Number of days with groups
    grpOb = np.sum(np.greater(grpsOb, 0), axis=1)

    # Average number of groups
    Gss_1D_ker = conv.Gaussian1DKernel(2)
    AvGrpOb = conv.convolve(np.nanmean(grpsOb, axis=1), Gss_1D_ker)

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

    # Append zeros to clamp area
    pltyOb = np.insert(pltyOb, 0, 0)
    pltyOb = np.append(pltyOb, 0)

    pltyGr = np.insert(pltyGr, 0, 0)
    pltyGr = np.append(pltyGr, 0)

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
    ppadh = padh / fszv  # Horizontal padding the edge of the figure in relative units

    # Start Figure
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

            if np.isnan(ssn_data.vldIntr[Idx]):
                ax3.plot([ssn_data.endPoints['OBS'][Idx, 0], ssn_data.endPoints['OBS'][Idx + 1, 0]], [0, np.ceil(np.nanmax(AvGrpOb)) + 1], color='r',
                         linestyle='--')
                ax3.plot([ssn_data.endPoints['OBS'][Idx + 1, 0], ssn_data.endPoints['OBS'][Idx, 0]], [0, np.ceil(np.nanmax(AvGrpOb)) + 1], color='r',
                         linestyle='--')

    ax3.plot((fyr1Ob + fyr2Ob) / 2, AvGrpOb, color=ssn_data.Clr[0], linewidth=2)

    # Axes properties
    ax3.set_xlabel('Year')
    ax3.set_ylabel('Average number of groups')
    ax3.set_xlim(left=np.min(fyr1Ob), right=np.max(fyr2Ob))
    ax3.grid(color=(0.5, 0.5, 0.5), linestyle='--', linewidth=1, axis='x', which='both')
    ax3.minorticks_on()
    ax3.set_ylim(bottom=0, top=np.ceil(np.nanmax(AvGrpOb)) + 1)

    fig.savefig(figure_path, bbox_inches='tight')

    fig.clear()
    plt.close(fig)

    print('done.', flush=True)
    print(' ', flush=True)


def plotOptimalThresholdWindow(ssn_data,
                               dpi=325,
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

    figure_path = config.get_file_output_string('05', 'Optimal_Threshold_Window',
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

    # Start Figure
    fig = plt.figure(figsize=(fszh / dpi, fszv / dpi), dpi=dpi)

    # Comparison with RGO
    ax2 = fig.add_axes([ppadh, ppadv, pxx / fszh, pxy / fszv])
    ax2.plot(ssn_data.REF_Grp['FRACYEAR'], ssn_data.REF_Grp['AVGROUPS'], 'r--', linewidth=2, alpha=1)

    # Plotting Observer
    ax2.plot(ssn_data.obsPlt['X'], ssn_data.obsPlt['Y'], color=ssn_data.Clr[0], linewidth=2)

    # Axes properties
    ax2.set_ylabel('Average Number of Groups')
    ax2.set_xlabel('Center of sliding window (Year)')
    ax2.set_xlim(left=np.min(ssn_data.REF_Grp['FRACYEAR']), right=np.max(ssn_data.REF_Grp['FRACYEAR']))
    ax2.set_ylim(bottom=0, top=np.max(ssn_data.REF_Grp['AVGROUPS']) * 1.1)

    # EMD Pcolor
    plt.viridis()

    # Creating Storing dictionaries to store best thresholds
    bestTh = []

    # Y positions to use in the mesh and plot
    y = np.array(ssn_data.Thresholds)

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

        # If we want to ignore overlaps
        if ssn_data.noOvrlpSw:
            Ovrlp = np.array(np.logical_and(ssn_data.REF_Dat['FRACYEAR'] >= ssn_data.endPoints['OBS'][siInx, 0],
                                            ssn_data.REF_Dat['FRACYEAR'] < ssn_data.endPoints['OBS'][siInx + 1, 0]).nonzero()[0])
            cadMaskI = np.setdiff1d(cadMaskI, Ovrlp)

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

        TObsSNd = ssn_data.ObsDat.loc[
            np.logical_and(ssn_data.ObsDat['FRACYEAR'] >= ssn_data.endPoints['OBS'][siInx, 0],
                           ssn_data.ObsDat['FRACYEAR'] < ssn_data.endPoints['OBS'][siInx + 1, 0])
            , 'AVGSNd'].values.copy()

        # Plot Matrix Only if the period is valid
        if ssn_data.vldIntr[siInx]:

            # Find index of minimum inside sub-interval
            minYear = np.min(np.absolute(TObsFYr - ssn_data.cenPoints['OBS'][siInx, 0]))
            obsMinInx = (np.absolute(TObsFYr - ssn_data.cenPoints['OBS'][siInx, 0]) == minYear).nonzero()[0][0]
            maxYear  = np.max(np.absolute(TObsFYr + ssn_data.cenPoints['OBS'][siInx, 0]))
            obsMaxInx = (np.absolute(TObsFYr + ssn_data.cenPoints['OBS'][siInx, 0]) == maxYear).nonzero()[0][0]

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
            if ((TObsFYr[obsMinInx] > np.min(ssn_data.REF_Dat['FRACYEAR'])) and (
                    TObsFYr[obsMinInx] < np.max(ssn_data.REF_Dat['FRACYEAR']))) and (
                (TObsFYr[obsMaxInx] > np.min(ssn_data.REF_Dat['FRACYEAR'])) and (
                    TObsFYr[obsMaxInx] < np.max(ssn_data.REF_Dat['FRACYEAR']))):

                # Check if first element is present in reference
                if np.any(ssn_data.REF_Dat['ORDINAL'] == TObsOrd[0]):

                    # Selecting the maximum integer amount of "months" out of the original data
                    TgrpsOb = TObsDat[0:np.int(TObsDat.shape[0] / ssn_data.MoLngt) * ssn_data.MoLngt].copy()

                    TObsSNd = TObsSNd[0:np.int(TObsDat.shape[0] / ssn_data.MoLngt) * ssn_data.MoLngt].copy()

                    # Calculating bracketing indices
                    Idx1 = (ssn_data.REF_Dat['ORDINAL'] == TObsOrd[0]).nonzero()[0][0]
                    Idx2 = Idx1 + TgrpsOb.shape[0]

                    TSNdREF = ssn_data.REF_Grp['AVGSNd'][Idx1:Idx2].values.copy()
                    TSNdREF = TSNdREF.reshape((-1, ssn_data.MoLngt))

                    # Going through different thresholds
                    for TIdx, Thr in enumerate(ssn_data.Thresholds):

                        if config.DEN_TYPE == 'DTh':
                            # Final fit to define threshold
                            highth = ssn_data.a1high * Thr + ssn_data.a0high
                            if Thr >= ssn_data.minVldThr:
                                lowth = ssn_data.a1low * Thr + ssn_data.a0low
                            else:
                                lowth = 0

                        # Calculating number of groups in reference data for given threshold
                        grpsREFw = np.nansum(
                            np.greater(ssn_data.REF_Dat.values[:, 3:ssn_data.REF_Dat.values.shape[1] - 3],
                                       Thr),
                            axis=1).astype(float)
                        grpsREFw[np.isnan(ssn_data.REF_Dat['AREA1'])] = np.nan

                        # Selecting the maximum integer amount of "months" out of the original data
                        TgrpsOb = TObsDat[0:np.int(TObsDat.shape[0] / ssn_data.MoLngt) * ssn_data.MoLngt].copy()

                        TObsSNd = TObsSNd[0:np.int(TObsDat.shape[0] / ssn_data.MoLngt) * ssn_data.MoLngt].copy()

                        # Selecting reference window of matching size to observer sub-interval;
                        TgrpsREF = grpsREFw[Idx1:Idx2].copy()

                        # Reshaping into "months"
                        TgrpsOb = TgrpsOb.reshape((-1, ssn_data.MoLngt))
                        TgrpsREF = TgrpsREF.reshape((-1, ssn_data.MoLngt))

                        TObsSNd = TObsSNd.reshape((-1, ssn_data.MoLngt))
                        TSNdREF = TSNdREF.reshape((-1, ssn_data.MoLngt))

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

                        # Number of quiet days
                        # OBSERVER
                        QDObsT = ODObsT - GDObsT
                        # REFERENCE
                        QDREFT = ODREFT - GDREFT

                        # Monthly sunspot number
                        SNdObsT = np.mean(TObsSNd, axis=1)
                        SNdREFT = np.mean(TSNdREF, axis=1)

                        # Numerator and denominator for given observer
                        numADObsT = GDObsT
                        numQDObsT = ssn_data.MoLngt - QDObsT
                        denFMObsT = GDObsT * 0 + ssn_data.MoLngt
                        denODObsT = ODObsT

                        # Numerator and denominator for reference
                        numADREFT = GDREFT
                        numQDREFT = ssn_data.MoLngt - QDREFT
                        denFMREFT = GDREFT * 0 + ssn_data.MoLngt
                        denODREFT = ODREFT

                        if config.NUM_TYPE == "ADF":
                            numObsT = numADObsT
                            numREFT = numADREFT
                        else:
                            numObsT = numQDObsT
                            numREFT = numQDREFT

                        if config.DEN_TYPE == "OBS":
                            denObsT = denODObsT
                            denREFT = denODREFT
                        else:
                            denObsT = denFMObsT
                            denREFT = denFMREFT

                        if config.DEN_TYPE == "DTh":
                            # defining solar activity level
                            MMObsT = np.logical_and((SNdObsT >= lowth), (SNdObsT < highth))
                            MMREFT = np.logical_and((SNdREFT >= lowth), (SNdREFT < highth))

                            HMObsT = (SNdObsT >= highth)
                            HMREFT = (SNdREFT >= highth)

                            numObsT[HMObsT] = numQDObsT[HMObsT]
                            numREFT[HMREFT] = numQDREFT[HMREFT]

                            denObsT[MMObsT] = denODObsT[MMObsT]
                            denREFT[MMREFT] = denODREFT[MMREFT]

                        ADF_Obs_fracT = np.divide(numObsT, denObsT)
                        ADF_REF_fracT = np.divide(numREFT, denREFT)

                        # Calculating Earth Mover's Distance
                        ADFObsDis, bins = np.histogram(
                            ADF_Obs_fracT[ODObsT / ssn_data.MoLngt >= ssn_data.minObD],
                            bins=ssn_data.EMDbins, density=True)

                        ADFREFDis, bins = np.histogram(
                            ADF_REF_fracT[ODREFT / ssn_data.MoLngt >= ssn_data.minObD],
                            bins=ssn_data.EMDbins, density=True)

                        tmp = emd(ADFREFDis.astype(np.float64), ADFObsDis.astype(np.float64),
                                  ssn_data.Dis.astype(np.float64))

                        # Udating variables
                        if tmp < tmpEMD:
                            tmpEMD = tmp
                            tmpt = TObsFYr[obsMinInx]
                            tmpth = Thr

            OpMat = np.insert(OpMat, 0, [tmpt, tmpth, tmpEMD], axis=0)

            # Calculating mesh for plotting
            x = ssn_data.REF_Grp['FRACYEAR'].values[cadMaskI]
            xx, yy = np.meshgrid(x, y)

            # Plotting Optimization Matrix
            ax1.pcolormesh(xx, yy, ssn_data.EMDD[siInx], alpha=1, linewidth=2, vmin=np.min(ssn_data.EMDD[siInx]),
                           vmax=6 * np.min(ssn_data.EMDD[siInx]))

            # True Interval
            ax1.scatter(OpMat[0, 0], OpMat[0, 1], c='r', edgecolors='w', linewidths=2, s=250, zorder=11)

            # Best point
            ax1.scatter(OpMat[1, 0], OpMat[1, 1], c='w', linewidths=2, s=200, zorder=11, alpha=0.75)
            ax2.plot(
                ssn_data.obsPlt['X'][
                    np.logical_and(ssn_data.obsPlt['X'] >= np.min(TObsFYr),
                                   ssn_data.obsPlt['X'] < np.max(TObsFYr))] -
                ssn_data.cenPoints['OBS'][siInx, 0] +
                OpMat[1, 0]
                , ssn_data.obsPlt['Y'][
                    np.logical_and(ssn_data.obsPlt['X'] >= np.min(TObsFYr),
                                   ssn_data.obsPlt['X'] < np.max(TObsFYr))],
                color=ssn_data.Clr[5 - siInx%6], linewidth=3
                , alpha=0.2)

            # Best 5 points
            if config.NBEST >= 5:
                for i in range(2, 6):
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
                        color=ssn_data.Clr[5 - siInx%6], linewidth=3
                        , alpha=0.2)

            # Best 5-10 points
            if config.NBEST >= 10:
                for i in range(6, 11):
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
                        color=ssn_data.Clr[5 - siInx%6], linewidth=3
                        , alpha=0.2)

            # Best 10-15 points
            if config.NBEST >= 15:
                for i in range(11, 16):
                    ax1.scatter(OpMat[i, 0], OpMat[i, 1], c='w', linewidths=2, s=50, zorder=11, alpha=0.5)

            # Best 15-20 points
            if config.NBEST >= 15:
                for i in range(16, 26):
                    ax1.scatter(OpMat[i, 0], OpMat[i, 1], c='w', linewidths=2, s=1, zorder=11, alpha=0.5)

            # Masking Gaps
            pltMsk = np.logical_not(cadMask)
            ax1.fill_between(ssn_data.REF_Dat['FRACYEAR'], ssn_data.REF_Dat['FRACYEAR'] * 0,
                             y2=ssn_data.REF_Dat['FRACYEAR'] * 0 + ssn_data.thE,
                             where=pltMsk, color='w', zorder=10)

            # Adding best points
            bestTh.append(OpMat[1:config.NBEST + 1, :])

            print(OpMat[1:config.NBEST + 1, :])
        # If period not valid store an empty array
        else:
            bestTh.append([])

        # Only plot real location if interval exists
        if ssn_data.vldIntr[siInx]:
            # Plotting real location
            ax1.plot(np.array([1, 1]) * TObsFYr[obsMinInx], np.array([0, np.max(y)]), 'w--', linewidth=3)

        # Plotting edges
        ax1.plot(np.array([1, 1]) * np.min(TObsFYr), np.array([0, np.max(y)]), ':', zorder=11, linewidth=3,
                 color=ssn_data.Clr[5 - siInx%6])
        ax1.plot(np.array([1, 1]) * np.max(TObsFYr), np.array([0, np.max(y)]), ':', zorder=11, linewidth=3,
                 color=ssn_data.Clr[5 - siInx%6])

        ax2.plot(np.array([1, 1]) * np.min(TObsFYr), np.array([0, np.max(y)]), ':', zorder=11, linewidth=3,
                 color=ssn_data.Clr[5 - siInx%6])
        ax2.plot(np.array([1, 1]) * np.max(TObsFYr), np.array([0, np.max(y)]), '-', zorder=11, linewidth=1,
                 color=ssn_data.Clr[5 - siInx%6])
        ax2.plot(np.array([1, 1]) * np.max(TObsFYr), np.array([0, np.max(y)]), ':', zorder=11, linewidth=3,
                 color=ssn_data.Clr[5 - siInx%6])

        # Axes properties
        ax1.set_ylabel('Area threshold (uHem)')
        ax1.set_xlim(left=np.min(ssn_data.REF_Dat['FRACYEAR']), right=np.max(ssn_data.REF_Dat['FRACYEAR']))
        ax1.set_ylim(bottom=ssn_data.thS, top=ssn_data.thE)

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

    fig.clear()
    plt.close(fig)

    print('done.', flush=True)
    print(' ', flush=True)


def plotDistributionOfThresholdsMI(ssn_data,
                                   dpi=325,
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

    figure_path = config.get_file_output_string('06', 'Distribution_of_Thresholds_MI',
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

    # Start Figure
    fig = plt.figure(figsize=(fszh / dpi, fszv / dpi), dpi=dpi)
    for i in range(0, nph):
        for j in range(0, npv):

            n = (nph * j + i)

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
                             density=True)

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
                             orientation='horizontal', density=True)

                    # # Axes properties
                    ax2.set_ylim(bottom=ssn_data.thS, top=ssn_data.thE)
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
                    ax1.set_ylim(bottom=ssn_data.thS, top=ssn_data.thE)

    fig.savefig(figure_path, bbox_inches='tight')

    fig.clear()
    plt.close(fig)

    print('done.', flush=True)
    print(' ', flush=True)


def plotHistSqrtSSN(ssn_data, ax, calRefT, calObsT, Th):
    """

    :param ax: Axis handle to plot the 2D histogram
    :param calRefT: Group counts for reference observed after application of threshold
    :param calObsT: Group counts of calibrated observer
    :param Th: Optimal threshold used to obtain data in this plot.  It is printed at the top of the figure
    """

    # Number of bins to use
    maxN = ssn_data.maxNPlt
    Nbins = ssn_data.maxNPlt

    # Average group number
    ax.hist2d(calObsT, calRefT, bins=ssn_data.edges, cmap=plt.cm.magma_r, cmin=1)

    # Calculating Quantities for plot
    hist, xedges, yedges = np.histogram2d(calObsT, calRefT, bins=ssn_data.edges)
    alphaY = np.sum(hist, axis=1) / np.max(np.sum(hist, axis=1)) * 0.95 + 0.05

    # Calculating Quantities for plot and plot centers in Y
    Ymedian = ssn_data.centers * np.nan
    for i in range(0, ssn_data.centers.shape[0]):
        ypoints = calRefT[np.logical_and(calObsT >= ssn_data.edges[i], calObsT < ssn_data.edges[i + 1])]
        if ypoints.shape[0] > 0:
            Ymedian[i] = np.nanmedian(ypoints)
            pecentilesy = np.abs(np.percentile(ypoints, np.array([15, 85]), interpolation='linear') - Ymedian[i])

            xpoints = calObsT[np.logical_and(calRefT >= (Ymedian[i] - (np.ceil(maxN)) / Nbins / 2),
                                             calRefT <= (Ymedian[i] + (np.ceil(maxN)) / Nbins / 2))]

            if xpoints.shape[0] > 0:
                pecentilesx = np.abs(
                    np.percentile(xpoints, np.array([15, 85]), interpolation='linear') - ssn_data.centers[i])
            else:
                pecentilesx = pecentilesy * 0

            ax.errorbar(ssn_data.centers[i], Ymedian[i], yerr=np.expand_dims(pecentilesy, axis=1),
                        xerr=np.expand_dims(pecentilesx, axis=1), color='k', zorder=11, alpha=alphaY[i])
            ax.scatter(ssn_data.centers[i], Ymedian[i], color='w', edgecolor='k', s=100, linewidths=3, zorder=11,
                       alpha=alphaY[i])

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

    # Calculating quantities for assessment
    y = Ymedian
    x = ssn_data.centers

    x = x[np.isfinite(y)]
    y = y[np.isfinite(y)]

    # R squared of the median
    yMean = np.mean(y)
    SStot = np.sum(np.power(y - yMean, 2))
    SSreg = np.sum(np.power(y - x, 2))
    rSqM = (1 - SSreg / SStot)

    # Mean Residual
    mResM = np.mean(y - x)
    # Mean Relative Residual
    mRResM = np.mean(np.divide(y[x > 0] - x[x > 0], x[x > 0]))

    ax.plot(ssn_data.edges, ssn_data.edges, '--'
            , color=ssn_data.Clr[4], linewidth=3)

    ax.text(0.5, 0.95,
            '$R^2$=' + str(np.round(rSq, decimals=2)) + ' $MR$=' + str(np.round(mRes, decimals=2)) + ' $MRR$=' + str(
                int(np.round(mRRes * 100, decimals=0))) + '%' +
            '\n$R^2M$=' + str(np.round(rSqM, decimals=2)) + ' $MRM$=' + str(
                np.round(mResM, decimals=2)) + ' $MRRM$=' + str(
                int(np.round(mRResM * 100, decimals=0))) + '% $Th$=' + str(
                Th), horizontalalignment='center', verticalalignment='center', transform=ax.transAxes)

    # Axes properties

    if config.SQRT_2DHIS:
        ax.set_xlabel('sqrt(GN+1) for ' + ssn_data.NamObs.capitalize())
        ax.set_ylabel('sqrt(GN+1) for reference')
    else:
        ax.set_xlabel('GN for ' + ssn_data.NamObs.capitalize())
        ax.set_ylabel('GN for reference observer')

    ax.set_xticks(np.arange(ssn_data.centers[0], np.ceil(ssn_data.centers[-1]),
                            np.floor((np.ceil(ssn_data.centers[-1]) - ssn_data.centers[0]) / 8) + 1))
    ax.set_yticks(np.arange(ssn_data.centers[0], np.ceil(ssn_data.centers[-1]),
                            np.floor((np.ceil(ssn_data.centers[-1]) - ssn_data.centers[0]) / 8) + 1))


def plotIntervalScatterPlots(ssn_data,
                             dpi=325,
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

    figure_path = config.get_file_output_string('07', 'Interval_Scatter_Plots',
                                                ssn_data=ssn_data,
                                                num_type=config.NUM_TYPE,
                                                den_type=config.DEN_TYPE)

    if config.SKIP_PRESENT_PLOTS and os.path.exists(figure_path):
        print("\nFigure at {} already exists.\n"
              " Change the OVERWRITE_OBSERVERS config flag to overwrite existing plots\n".format(
            figure_path))
        return

    calRef = np.array([0])
    calObs = np.array([0])

    frc = 1  # Fraction of the panel devoted to histograms

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

    # Start Figure
    fig = plt.figure(figsize=(fszh / dpi, fszv / dpi), dpi=dpi)

    # Calculating maximum for plotting
    maxNPlt = 0
    # Going through different sub-intervals
    for siInx in range(0, ssn_data.cenPoints['OBS'].shape[0]):
        # Perform analysis Only if the period is valid
        if ssn_data.vldIntr[siInx] and np.sum(
                np.logical_and(ssn_data.REF_Dat['FRACYEAR'] >= ssn_data.endPoints['OBS'][siInx, 0],
                               ssn_data.REF_Dat['FRACYEAR'] < ssn_data.endPoints['OBS'][siInx + 1, 0])) > 0:
            TObsDat = ssn_data.ObsDat.loc[
                np.logical_and(ssn_data.ObsDat['FRACYEAR'] >= ssn_data.endPoints['OBS'][siInx, 0],
                               ssn_data.ObsDat['FRACYEAR'] < ssn_data.endPoints['OBS'][siInx + 1, 0])
                , 'GROUPS'].values.copy()
            TObsREFDat = np.nansum(
                np.greater(ssn_data.REF_Dat.values[:, 3:ssn_data.REF_Dat.values.shape[1] - 3], ssn_data.wAvI[siInx]),
                axis=1).astype(float)
            TObsREFDat = TObsREFDat[np.logical_and(ssn_data.REF_Dat['FRACYEAR'] >= ssn_data.endPoints['OBS'][siInx, 0],
                                                   ssn_data.REF_Dat['FRACYEAR'] < ssn_data.endPoints['OBS'][
                                                       siInx + 1, 0])].copy()
            maxNPlt = np.max([np.nanmax(TObsREFDat), np.nanmax(TObsDat), maxNPlt])

    for i in range(0, nph):
        for j in range(0, npv):

            n = (nph * j + i)

            # Only add the panel if it exists
            if n < ssn_data.vldIntr.shape[0]:

                # Plot only if the period is valid and has overlap
                if ssn_data.vldIntr[n] and np.sum(
                        np.logical_and(ssn_data.REF_Dat['FRACYEAR'] >= ssn_data.endPoints['OBS'][n, 0],
                                       ssn_data.REF_Dat['FRACYEAR'] < ssn_data.endPoints['OBS'][
                                           n + 1, 0])) > 0:
                    # Calculating number of groups in reference data for given threshold
                    grpsREFw = np.nansum(np.greater(ssn_data.REF_Dat.values[:, 3:ssn_data.REF_Dat.values.shape[1] - 3],
                                                    ssn_data.wAvI[n]), axis=1).astype(float)
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
                    calRef = np.append(calRef, grpsREFw)
                    calObs = np.append(calObs, grpsObsw)

                    # Average group number
                    ax1 = fig.add_axes(
                        [ppadh + i * (pxx / fszh + ppadh2), ppadv + j * (pxy / fszv + ppadv2), pxx / fszh * frc,
                         pxy / fszv * frc], label='b' + str(n))

                    plotHistSqrtSSN(ssn_data, ax1, grpsREFw, grpsObsw, np.round(ssn_data.wAvI[n], decimals=1))

                    ax1.text(0.5, 0.87,
                             'From ' + str(np.round(ssn_data.endPoints['OBS'][n, 0], decimals=2)) + '  to ' + str(
                                 np.round(ssn_data.endPoints['OBS'][n + 1, 0], decimals=2)),
                             horizontalalignment='center',
                             verticalalignment='center', transform=ax1.transAxes)

    fig.savefig(figure_path, bbox_inches='tight')

    fig.clear()
    plt.close(fig)

    print('done.', flush=True)
    print(' ', flush=True)


def _histOutline(histIn, binsIn):
    """
    Wrapper around the histogram to create an outline that can be used to plot distributions

    :param dataIn:
    :param args:
    :param kwargs:
    :return:
    """
    #
    # (histIn, binsIn) = np.histogram(dataIn, *args, **kwargs)

    stepSize = binsIn[1] - binsIn[0]

    bins = np.zeros(len(binsIn) * 2 + 2, dtype=np.float)
    data = np.zeros(len(binsIn) * 2 + 2, dtype=np.float)
    for bb in range(len(binsIn)):
        bins[2 * bb + 1] = binsIn[bb]
        bins[2 * bb + 2] = binsIn[bb] + stepSize
        if bb < len(histIn):
            data[2 * bb + 1] = histIn[bb]
            data[2 * bb + 2] = histIn[bb]

    bins[0] = bins[1]
    bins[-1] = bins[-2]
    data[0] = 0
    data[-1] = 0

    return bins, data


def plotIntervalDistributions(ssn_data,
                              dpi=325,
                              pxx=2300,
                              pxy=1000,
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

    print('Creating and saving interval distribution-plots figure...', end="", flush=True)

    figure_path = config.get_file_output_string('08', 'Interval_Distribution_Plots',
                                                ssn_data=ssn_data,
                                                num_type=config.NUM_TYPE,
                                                den_type=config.DEN_TYPE)

    if config.SKIP_PRESENT_PLOTS and os.path.exists(figure_path):
        print("\nFigure at {} already exists.\n"
              " Change the OVERWRITE_OBSERVERS config flag to overwrite existing plots\n".format(
            figure_path))
        return

    frc = 1  # Fraction of the panel devoted to histograms

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

    # Start Figure
    fig = plt.figure(figsize=(fszh / dpi, fszv / dpi), dpi=dpi)
    for i in range(0, nph):
        for j in range(0, npv):

            siInx = (nph * j + i)

            # Only add the panel if it exists
            if siInx < ssn_data.vldIntr.shape[0]:

                # Plot only if the period is valid
                if ssn_data.vldIntr[siInx]:
                    SIdx = int(ssn_data.bestTh[siInx][0][3])
                    TIdx = int(ssn_data.bestTh[siInx][0][4])

                    ax1 = fig.add_axes(
                        [ppadh + i * (pxx / fszh + ppadh2), ppadv + j * (pxy / fszv + ppadv2), pxx / fszh * frc,
                         pxy / fszv * frc], label='b' + str(siInx))

                    (xAD, yAD) = _histOutline(ssn_data.ADFObsMD[siInx][TIdx, SIdx, :], ssn_data.EMDbins)
                    ax1.fill_between(xAD, yAD, color='0.5')

                    (xAD, yAD) = _histOutline(ssn_data.ADFRefMD[siInx][TIdx, SIdx, :], ssn_data.EMDbins)
                    ax1.plot(xAD, yAD, color=ssn_data.Clr[4], linewidth=3)

                    ax1.text(0.02, 0.96, 'From ' + str(np.round(ssn_data.endPoints['OBS'][siInx, 0], decimals=1)) + ' to '
                             + str(np.round(ssn_data.endPoints['OBS'][siInx + 1, 0], decimals=1))
                             + '\nbest match: ' + str(np.round(ssn_data.bestTh[siInx][0][0], decimals=1))
                             + '  Th: ' + str(int(ssn_data.bestTh[siInx][0][1])), horizontalalignment='left',
                             verticalalignment='top', transform=ax1.transAxes)
                    ax1.set_xlabel('ADF')
                    ax1.set_ylabel('PDF')
                    ax1.set_xlim(left=np.min(ssn_data.EMDbins), right=np.max(ssn_data.EMDbins))
                    ax1.set_ylim(bottom=0)

    fig.savefig(figure_path, bbox_inches='tight')

    fig.clear()
    plt.close(fig)

    print('done.', flush=True)
    print(' ', flush=True)


def plotMinEMD(ssn_data,
               dpi=325,
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

    figure_path = config.get_file_output_string('09', 'Min_EMD',
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

    # Start Figure
    fig = plt.figure(figsize=(fszh / dpi, fszv / dpi), dpi=dpi)

    # Comparison with RGO
    ax2 = fig.add_axes([ppadh, ppadv, pxx / fszh, pxy / fszv])
    ax2.plot(ssn_data.REF_Grp['FRACYEAR'], ssn_data.REF_Grp['AVGROUPS'], 'r--', linewidth=2, alpha=1)

    # Plotting Observer
    ax2.plot(ssn_data.obsPlt['X'], ssn_data.obsPlt['Y'], color=ssn_data.Clr[0], linewidth=2)

    # Axes properties
    ax2.set_ylabel('Average Number of Groups')
    ax2.set_xlabel('Center of sliding window (Year)')
    ax2.set_xlim(left=np.min(ssn_data.REF_Grp['FRACYEAR']), right=np.max(ssn_data.REF_Grp['FRACYEAR']))
    ax2.set_ylim(bottom=0, top=np.max(ssn_data.REF_Grp['AVGROUPS']) * 1.1)

    # Initialize y to avoid crash when the first interval is invalid
    y = 1
    # Going through different sub-intervals
    for siInx in range(0, ssn_data.cenPoints['OBS'].shape[0]):

        # Defining mask based on the interval type (rise or decay)
        if ssn_data.cenPoints['OBS'][siInx, 1] > 0:
            cadMaskI = ssn_data.risMask['INDEX']
            cadMask = ssn_data.risMask['PLOT']
        else:
            cadMaskI = ssn_data.decMask['INDEX']
            cadMask = ssn_data.decMask['PLOT']

        # If we want to ignore overlaps
        if ssn_data.noOvrlpSw:
            Ovrlp = np.array(np.logical_and(ssn_data.REF_Dat['FRACYEAR'] >= ssn_data.endPoints['OBS'][siInx, 0],
                                            ssn_data.REF_Dat['FRACYEAR'] < ssn_data.endPoints['OBS'][
                                                siInx + 1, 0]).nonzero()[0])
            cadMaskI = np.setdiff1d(cadMaskI, Ovrlp)


        # Creating axis
        ax1 = fig.add_axes([ppadh, ppadv + (siInx + 1) * (pxy / fszv + ppadv2), pxx / fszh, pxy / fszv])

        TObsFYr = ssn_data.ObsDat.loc[
            np.logical_and(ssn_data.ObsDat['FRACYEAR'] >= ssn_data.endPoints['OBS'][siInx, 0],
                           ssn_data.ObsDat['FRACYEAR'] < ssn_data.endPoints['OBS'][siInx + 1, 0])
            , 'FRACYEAR'].values.copy()

        # Plot only if period is valid
        if ssn_data.vldIntr[siInx]:
            # Calculating minimum distance for plotting
            x = ssn_data.REF_Grp['FRACYEAR'].values[cadMaskI]
            y = np.amin(ssn_data.EMDD[siInx], axis=0)
            sortIn = np.argsort(y)

            # Plotting Optimization Matrix
            ax1.plot(x, y, color='k', linewidth=3)
            ax1.scatter(x[sortIn[0:ssn_data.NTshifts]], y[sortIn[0:ssn_data.NTshifts]], color='r', s=100, zorder=11)

            # Masking Gaps
            pltMsk = np.logical_not(cadMask)
            ax1.fill_between(ssn_data.REF_Grp['FRACYEAR'], ssn_data.REF_Grp['FRACYEAR'] * 0,
                             y2=(ssn_data.REF_Grp['FRACYEAR'] * 0 + np.min(y) * 10 + 20), where=pltMsk, color='w',
                             zorder=10)  # y2 is the color intensity of the border for earch panel. We added "+20" to get a thin line for the panel according to the observer #563 where min(y) is equal to 0

        # Plotting edges
        ax1.plot(np.array([1, 1]) * np.min(TObsFYr), np.array([0, np.max(y)]), ':', zorder=11, linewidth=3,
                 color=ssn_data.Clr[5 - siInx%6])
        ax1.plot(np.array([1, 1]) * np.max(TObsFYr), np.array([0, np.max(y)]), ':', zorder=11, linewidth=3,
                 color=ssn_data.Clr[5 - siInx%6])

        ax2.plot(np.array([1, 1]) * np.min(TObsFYr), np.array([0, np.max(y)]), ':', zorder=11, linewidth=3,
                 color=ssn_data.Clr[5 - siInx%6])
        ax2.plot(np.array([1, 1]) * np.max(TObsFYr), np.array([0, np.max(y)]), '-', zorder=11, linewidth=1,
                 color=ssn_data.Clr[5 - siInx%6])
        ax2.plot(np.array([1, 1]) * np.max(TObsFYr), np.array([0, np.max(y)]), ':', zorder=11, linewidth=3,
                 color=ssn_data.Clr[5 - siInx%6])

        # Axes properties
        ax1.set_ylabel('Distribution Distance')
        ax1.set_xlim(left=np.min(ssn_data.REF_Dat['FRACYEAR']), right=np.max(ssn_data.REF_Dat['FRACYEAR']))
        ax1.set_ylim(bottom=0, top=np.nanmin(y) * 5 + 10)

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

    fig.clear()
    plt.close(fig)

    print('done.', flush=True)
    print(' ', flush=True)


def plotSimultaneousFit(ssn_data,
                        dpi=325,
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

    figure_path = config.get_file_output_string('10', 'Simultaneous_Fit',
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

    # Fraction of the panel devoted to histogram
    if config.NBEST == 1:
        frc = 1.0
    else:
        frc = 0.9

    nph = 1  # Number of horizontal panels
    npv = 2  # Number of vertical panels

    # Figure sizes in pixels
    fszv = (npv * pxy + 2 * padv + (npv - 1) * padv2)  # Vertical size of figure in inches
    fszh = (nph * pxx + 2 * padh + (nph - 1) * padh2)  # Horizontal size of figure in inches

    # Conversion to relative unites
    ppadv = padv / fszv  # Vertical padding in relative units
    ppadv2 = padv2 / fszv  # Vertical padding in relative units
    ppadh = padh / fszv  # Horizontal padding the edge of the figure in relative units

    ## Start Figure
    fig = plt.figure(figsize=(fszh / dpi, fszv / dpi), dpi=dpi)

    # Comparison with RGO
    ax2 = fig.add_axes([ppadh, ppadv, pxx / fszh * frc, pxy / fszv])
    ax2.plot(ssn_data.REF_Grp['FRACYEAR'], ssn_data.REF_Grp['AVGROUPS'], 'r--', linewidth=2, alpha=1)

    # Plotting Observer
    ax2.plot(ssn_data.obsPlt['X'], ssn_data.obsPlt['Y'], color=ssn_data.Clr[0], linewidth=2)

    # Axes properties
    ax2.set_ylabel('Av. Num. of Groups')
    ax2.set_xlabel('Center of sliding window (Year)')
    ax2.set_xlim(left=np.min(ssn_data.REF_Grp['FRACYEAR']), right=np.max(ssn_data.REF_Grp['FRACYEAR']))
    ax2.set_ylim(bottom=0, top=np.max(ssn_data.REF_Grp['AVGROUPS']) * 1.1)

    # Placement of top simultaneous fits
    ax1 = fig.add_axes([ppadh, ppadv + (pxy / fszv + ppadv2), pxx / fszh * frc, pxy / fszv])

    # Going through different sub-intervals
    for siInx in range(0, ssn_data.cenPoints['OBS'].shape[0]):
        TObsFYr = ssn_data.ObsDat.loc[
            np.logical_and(ssn_data.ObsDat['FRACYEAR'] >= ssn_data.endPoints['OBS'][siInx, 0],
                           ssn_data.ObsDat['FRACYEAR'] < ssn_data.endPoints['OBS'][siInx + 1, 0])
            , 'FRACYEAR'].values.copy()

        # Plotting edges
        ax1.plot(np.array([1, 1]) * np.min(TObsFYr), np.array([0, ssn_data.thE]), ':', zorder=11,
                 linewidth=3,
                 color=ssn_data.Clr[5 - siInx%6])
        ax1.plot(np.array([1, 1]) * np.max(TObsFYr), np.array([0, ssn_data.thE]), '-', zorder=11,
                 linewidth=1,
                 color=ssn_data.Clr[5 - siInx%6])
        ax1.plot(np.array([1, 1]) * np.max(TObsFYr), np.array([0, ssn_data.thE]), ':', zorder=11,
                 linewidth=3,
                 color=ssn_data.Clr[5 - siInx%6])

        ax2.plot(np.array([1, 1]) * np.min(TObsFYr), np.array([0, ssn_data.thE]), ':', zorder=11,
                 linewidth=3,
                 color=ssn_data.Clr[5 - siInx%6])
        ax2.plot(np.array([1, 1]) * np.max(TObsFYr), np.array([0, ssn_data.thE]), '-', zorder=11,
                 linewidth=1,
                 color=ssn_data.Clr[5 - siInx%6])
        ax2.plot(np.array([1, 1]) * np.max(TObsFYr), np.array([0, ssn_data.thE]), ':', zorder=11,
                 linewidth=3,
                 color=ssn_data.Clr[5 - siInx%6])

    for i in range(0, config.NBEST):

        # Initialize plot vector
        x = np.array([])

        for siInx in range(0, ssn_data.cenPoints['OBS'].shape[0]):

            # Defining mask based on the interval type (rise or decay)
            if ssn_data.cenPoints['OBS'][siInx, 1] > 0:
                cadMaskI = ssn_data.risMask['INDEX']
            else:
                cadMaskI = ssn_data.decMask['INDEX']

            # If we want to ignore overlaps
            if ssn_data.noOvrlpSw:
                Ovrlp = np.array(np.logical_and(ssn_data.REF_Dat['FRACYEAR'] >= ssn_data.endPoints['OBS'][siInx, 0],
                                                ssn_data.REF_Dat['FRACYEAR'] < ssn_data.endPoints['OBS'][siInx + 1, 0]).to_numpy().nonzero()[0])
                cadMaskI = np.setdiff1d(cadMaskI, Ovrlp)

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

        # Only plot if using more than one theshold
        if config.NBEST == 1:
            alph = 1
        else:
            # Constructing alpha
            alph = np.clip(
                1.1 - (ssn_data.EMDComb[0, i] - np.min(ssn_data.EMDComb[0, :])) / (
                        np.max(ssn_data.EMDComb[0, :]) - np.min(ssn_data.EMDComb[0, :])), 0, 1)

        # Plotting Intervals
        ax1.plot(x, y, 'o:', zorder=11, linewidth=1, color=ssn_data.Clr[2], alpha=alph, markersize=(101 - i) / 8)

    # Axes properties
    ax1.set_ylabel('Area thres. (uHem)')
    ax1.set_xlim(left=np.min(ssn_data.REF_Dat['FRACYEAR']), right=np.max(ssn_data.REF_Dat['FRACYEAR']))
    ax1.set_ylim(bottom=ssn_data.thS, top=ssn_data.thE)

    ax1.text(0.5, 1.01, 'Best Simultaneous Fits for ' + ssn_data.NamObs, horizontalalignment='center',
             transform=ax1.transAxes)

    if config.NBEST > 1:
        # Right Distribution
        ax3 = fig.add_axes(
            [ppadh + pxx / fszh * frc, ppadv + (pxy / fszv + ppadv2), pxx / fszh * (1 - frc), pxy / fszv])
        ax3.hist(ssn_data.EMDComb[1, :], bins=np.array(ssn_data.Thresholds) + ssn_data.thI / 2,
                 color=ssn_data.Clr[2], alpha=.6,
                 orientation='horizontal', density=True)
        # ax3.plot(yOD, xOD, color=Clr[2], linewidth=3)

        # # Axes properties
        ax3.set_ylim(bottom=ssn_data.thS, top=ssn_data.thE)
        ax3.set_axis_off()

    fig.savefig(figure_path, bbox_inches='tight')

    fig.clear()
    plt.close(fig)

    print('done.', flush=True)
    print(' ', flush=True)


def plotDistributionOfThresholds(ssn_data,
                                 dpi=325,
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

    figure_path = config.get_file_output_string('11', 'Distribution_of_Thresholds',
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
    ppadh = padh / fszv  # Horizontal padding the edge of the figure in relative units

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
             orientation='horizontal', density=True)

    # # Axes properties
    ax2.set_ylim(bottom=ssn_data.thS, top=ssn_data.thE)
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
    ax1.set_ylim(bottom=ssn_data.thS, top=ssn_data.thE)

    fig.savefig(figure_path, bbox_inches='tight')

    fig.clear()
    plt.close(fig)

    print('done.', flush=True)
    print(' ', flush=True)


def plotSingleThresholdDistributions(ssn_data,
                                     dpi=325,
                                     pxx=2300,
                                     pxy=1000,
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

    print('Creating and saving interval distribution-plots figure...', end="", flush=True)

    figure_path = config.get_file_output_string('12', 'Single_Threshold_Distribution_Plot',
                                                ssn_data=ssn_data,
                                                num_type=config.NUM_TYPE,
                                                den_type=config.DEN_TYPE)

    if config.SKIP_PRESENT_PLOTS and os.path.exists(figure_path):
        print("\nFigure at {} already exists.\n"
              " Change the OVERWRITE_OBSERVERS config flag to overwrite existing plots\n".format(
            figure_path))
        return

    frc = 1  # Fraction of the panel devoted to histograms

    nph = 1  # Number of horizontal panels
    npv = 1  # Number of vertical panels

    # Figure sizes in pixels
    fszv = (npv * pxy + 2 * padv + (npv - 1) * padv2)  # Vertical size of figure in inches
    fszh = (nph * pxx + 2 * padh + (nph - 1) * padh2)  # Horizontal size of figure in inches

    # Conversion to relative unites
    ppadv = padv / fszv  # Vertical padding in relative units
    ppadh = padh / fszv  # Horizontal padding the edge of the figure in relative units

    txt1 = 'Real Loc. ->  '
    txt2 = 'Optimal   ->  '

    # Joining ADF from all sub-interval for the specified shifts
    for siInx in range(0, ssn_data.cenPoints['OBS'].shape[0]):

        # Defining mask based on the interval type (rise or decay)
        if ssn_data.cenPoints['OBS'][siInx, 1] > 0:
            cadMaskI = ssn_data.risMask['INDEX']
        else:
            cadMaskI = ssn_data.decMask['INDEX']

        # If we want to ignore overlaps
        if ssn_data.noOvrlpSw:
            Ovrlp = np.array(np.logical_and(ssn_data.REF_Dat['FRACYEAR'] >= ssn_data.endPoints['OBS'][siInx, 0],
                                            ssn_data.REF_Dat['FRACYEAR'] < ssn_data.endPoints['OBS'][
                                                siInx + 1, 0]).to_numpy().nonzero()[0])
            cadMaskI = np.setdiff1d(cadMaskI, Ovrlp)

        Year = ssn_data.REF_Grp['FRACYEAR'].values[cadMaskI]

        # Append only if period is valid
        if ssn_data.vldIntr[siInx]:

            TObsFYr = ssn_data.ObsDat.loc[
                np.logical_and(ssn_data.ObsDat['FRACYEAR'] >= ssn_data.endPoints['OBS'][siInx, 0],
                               ssn_data.ObsDat['FRACYEAR'] < ssn_data.endPoints['OBS'][siInx + 1, 0])
                , 'FRACYEAR'].values.copy()

            txt1 = txt1 + str(siInx + 1) + ': ' + str(
                np.round((np.min(TObsFYr) + np.max(TObsFYr)) / 2, decimals=1)) + '  '
            txt2 = txt2 + str(siInx + 1) + ': ' + str(
                np.round(Year[ssn_data.EMDComb[siInx + 2, 0].astype(np.int)], decimals=1)) + '  '


    ADFObs = ssn_data.EMDComb[ssn_data.EMDComb.shape[0]-2*ssn_data.MoLngt-1:ssn_data.EMDComb.shape[0]-ssn_data.MoLngt-1, 0]
    ADFREf = ssn_data.EMDComb[ssn_data.EMDComb.shape[0]-ssn_data.MoLngt:ssn_data.EMDComb.shape[0], 0]

    # Start Figure
    fig = plt.figure(figsize=(fszh / dpi, fszv / dpi))

    ax1 = fig.add_axes([ppadh, ppadv, pxx / fszh * frc, pxy / fszv * frc])

    (xAD, yAD) = _histOutline(ADFObs, ssn_data.EMDbins)
    ax1.fill_between(xAD, yAD, color='0.5')

    (xAD, yAD) = _histOutline(ADFREf, ssn_data.EMDbins)
    ax1.plot(xAD, yAD, color=ssn_data.Clr[4], linewidth=3)

    ax1.text(0.015, 0.96, txt1 + '\n' + txt2 + '\nTh: ' + str(int(ssn_data.wAv)), horizontalalignment='left',
             verticalalignment='top', transform=ax1.transAxes)
    ax1.set_title('Single threshold - Best distribution match')
    ax1.set_xlabel('ADF')
    ax1.set_ylabel('PDF')

    ax1.set_xlim(left=np.min(ssn_data.EMDbins), right=np.max(ssn_data.EMDbins))
    ax1.set_ylim(bottom=0)

    fig.savefig(figure_path, bbox_inches='tight')

    fig.clear()
    plt.close(fig)

    print('done.', flush=True)
    print(' ', flush=True)


def plotSingleThresholdScatterPlot(ssn_data,
                                   dpi=325,
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

    figure_path = config.get_file_output_string('13', 'Single_Threshold_ScatterPlot',
                                                ssn_data=ssn_data,
                                                num_type=config.NUM_TYPE,
                                                den_type=config.DEN_TYPE)

    if config.SKIP_PRESENT_PLOTS and os.path.exists(figure_path):
        print(
            "\nFigure at {} already exists.\n Change the OVERWRITE_OBSERVERS config flag to overwrite existing plots\n".format(
                figure_path))
        return

    # Test if there is any overlap
    # if (np.min(ssn_data.REF_Dat['ORDINAL']) <= np.min(ssn_data.ObsDat['ORDINAL'])) or (
    #         np.max(ssn_data.REF_Dat['ORDINAL']) >= np.max(ssn_data.ObsDat['ORDINAL'])):
    if ((np.min(ssn_data.REF_Dat['ORDINAL']) <= np.max(ssn_data.ObsDat['ORDINAL'])) and (
            np.max(ssn_data.REF_Dat['ORDINAL']) >= np.max(ssn_data.ObsDat['ORDINAL']))) or (
            (np.max(ssn_data.REF_Dat['ORDINAL']) >= np.min(ssn_data.ObsDat['ORDINAL'])) and (
            np.min(ssn_data.REF_Dat['ORDINAL']) <= np.min(ssn_data.ObsDat['ORDINAL']))):
        # Calculating number of groups in reference data for given threshold
        grpsREFw = np.nansum(
            np.greater(ssn_data.REF_Dat.values[:, 3:ssn_data.REF_Dat.values.shape[1] - 3], ssn_data.wAv),
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

        nph = 1  # Number of horizontal panels
        npv = 1  # Number of vertical panels

        # Figure sizes in pixels
        fszv = (npv * pxy + 2 * padv + (npv - 1) * padv2)  # Vertical size of figure in inches
        fszh = (nph * pxx + 2 * padh + (nph - 1) * padh2)  # Horizontal size of figure in inches

        # Conversion to relative unites
        ppadv = padv / fszv  # Vertical padding in relative units
        ppadh = padh / fszv  # Horizontal padding the edge of the figure in relative units

        # Start Figure
        fig = plt.figure(figsize=(fszh / dpi, fszv / dpi))

        # Average group number
        ax1 = fig.add_axes([ppadh, ppadv, pxx / fszh, pxy / fszv])

        plotHistSqrtSSN(ssn_data, ax1, grpsREFw, grpsObsw, np.round(ssn_data.wAv, decimals=1))

        ax1.set_title('Single threshold - All days of overlap')

        fig.savefig(figure_path, bbox_inches='tight')

        fig.clear()
        plt.close(fig)

        print('done.', flush=True)
        print(' ', flush=True)


def plotMultiThresholdScatterPlot(ssn_data,
                                  dpi=325,
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

        figure_path = config.get_file_output_string('14', 'Multi_Threshold_ScatterPlot',
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

        # Figure sizes in pixels
        fszv = (npv * pxy + 2 * padv + (npv - 1) * padv2)  # Vertical size of figure in inches
        fszh = (nph * pxx + 2 * padh + (nph - 1) * padh2)  # Horizontal size of figure in inches

        # Conversion to relative unites
        ppadv = padv / fszv  # Vertical padding in relative units
        ppadh = padh / fszv  # Horizontal padding the edge of the figure in relative units
        ppadh2 = padh2 / fszv  # Horizontal padding between panels in relative units

        # Start Figure
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

        plotHistSqrtSSN(ssn_data, ax1, calRefN, calObsN, np.round(ssn_data.wAv, decimals=1))

        # Average group number
        ax2 = fig.add_axes([ppadh + (pxx / fszh + ppadh2), ppadv, pxx / fszh, pxy / fszv], label='b2')

        plotHistSqrtSSN(ssn_data, ax2, tcalRef, tcalObs, 'Variable')

    fig.savefig(figure_path, bbox_inches='tight')

    fig.clear()
    plt.close(fig)

    print('done.', flush=True)
    print(' ', flush=True)


def plotSmoothedSeries(ssn_data,
                       dpi=325,
                       pxx=4000,
                       pxy=1500,
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

    if tcalRef.shape[0] > 1:

        print('Creating and saving smoothed series comparing thresholded reference with observer...', end="",
              flush=True)

        figure_path = config.get_file_output_string('15', 'SmoothedSeriesPlot',
                                                    ssn_data=ssn_data,
                                                    num_type=config.NUM_TYPE,
                                                    den_type=config.DEN_TYPE)

        if config.SKIP_PRESENT_PLOTS and os.path.exists(figure_path):
            print("\nFigure at {} already exists.\n"
                  " Change the OVERWRITE_OBSERVERS config flag to overwrite existing plots\n".format(
                figure_path))
            return

        # Creating variables for plotting
        Grp_Comp = ssn_data.Grp_Comp
        maxplt = np.max(
            Grp_Comp.loc[np.in1d(ssn_data.REF_Dat['ORDINAL'].values, ssn_data.ObsDat['ORDINAL'].values), 'GROUPS'])

        nph = 1  # Number of horizontal panels
        npv = 2  # Number of vertical panels

        # Figure sizes in pixels
        fszv = (npv * pxy + 2 * padv + (npv - 1) * padv2)  # Vertical size of figure in inches
        fszh = (nph * pxx + 2 * padh + (nph - 1) * padh2)  # Horizontal size of figure in inches

        # Conversion to relative unites
        ppadv = padv / fszv  # Vertical padding in relative units
        ppadh = padh / fszv  # Horizontal padding the edge of the figure in relative units

        ## Start Figure
        fig = plt.figure(figsize=(fszh / dpi, fszv / dpi), dpi=dpi)
        ax2 = fig.add_axes([ppadh, ppadv + pxy / fszv, pxx / fszh, pxy / fszv])

        pltx = Grp_Comp['FRACYEAR']

        ax2.plot(pltx, Grp_Comp['GROUPS'], 'r--', linewidth=2, alpha=1)
        ax2.plot(pltx, Grp_Comp['SINGLETH'], 'k', linewidth=2, alpha=0.15)
        ax2.plot(pltx, Grp_Comp['SINGLETHVI'], 'k:', linewidth=4, alpha=1)

        ax2.plot(pltx, Grp_Comp['CALOBS'], color=ssn_data.Clr[4], linewidth=4, alpha=1)
        ax2.plot(pltx, Grp_Comp['SINGLETHreal'], 'g--', linewidth=2, alpha=1)

        ax2.set_xlim(left=np.min(ssn_data.ObsDat['FRACYEAR']) - 7, right=np.max(ssn_data.ObsDat['FRACYEAR']) + 7)
        ax2.set_ylim(bottom=0, top=maxplt * 1.25)
        ax2.xaxis.tick_top()
        ax2.set_ylabel('Average Number of Groups')

        ax2.text(0.5, 0.05, 'Th:' + str(np.round(ssn_data.wAv, decimals=1)), horizontalalignment='center',
                 verticalalignment='center', transform=ax2.transAxes)
        ax2.legend(
            ['Smoothed Ref. GN', 'Smoothed Ref. GN - Single Threshold', 'Smoothed Ref. GN - Single Threshold Valid',
             'Smoothed GN for ' + ssn_data.NamObs.capitalize(), 'Smoothed Ref. GN - Real Threshold'],
            loc='upper center', ncol=2, frameon=True, edgecolor='none', fontsize=18)

        MRE = np.round(np.nanmean(np.divide(Grp_Comp['SINGLETHVI'] - Grp_Comp['CALOBS'], Grp_Comp['CALOBS'])),
                              decimals=2)
        MREav = np.round(np.nanmean(Grp_Comp['SINGLETHVI'] - Grp_Comp['CALOBS']) / np.mean(Grp_Comp['CALOBS']),
                        decimals=2)
        slp = np.round(np.nanmean(Grp_Comp['SINGLETHVI'] / Grp_Comp['CALOBS']), decimals=2)

        # ax2.text(0.005, 0.2, 'MRE:' + str(MRE), horizontalalignment='left', verticalalignment='center',
        #          transform=ax2.transAxes)
        # ax2.text(0.005, 0.125, 'MNEav:' + str(MREav), horizontalalignment='left', verticalalignment='center',
        #          transform=ax2.transAxes)
        # ax2.text(0.005, 0.055, 'slp:' + str(slp), horizontalalignment='left', verticalalignment='center',
        #          transform=ax2.transAxes)
        ax2.text(0.005, 0.275, 'MRE:' + str(MRE), horizontalalignment='left', verticalalignment='center',
                 transform=ax2.transAxes)
        ax2.text(0.005, 0.2, 'MNEav:' + str(MREav), horizontalalignment='left', verticalalignment='center',
                 transform=ax2.transAxes)
        ax2.text(0.005, 0.125, 'slp:' + str(slp), horizontalalignment='left', verticalalignment='center',
                 transform=ax2.transAxes)
        ax2.text(0.005, 0.05, 'Real Threshold: ' + str(np.round(ssn_data.realThr, decimals=0)), horizontalalignment='left',
                 verticalalignment='center', transform=ax2.transAxes)

        ax1 = fig.add_axes([ppadh, ppadv, pxx / fszh, pxy / fszv])

        # Plotting Observer
        ax1.plot(pltx, Grp_Comp['GROUPS'], 'r--', linewidth=2, alpha=1)
        ax1.plot(pltx, Grp_Comp['SINGLETH'], 'k', linewidth=2, alpha=0.15)
        ax1.plot(pltx, Grp_Comp['MULTITH'], color=ssn_data.Clr[2], linestyle=':', linewidth=4, alpha=1)

        ax1.plot(pltx, Grp_Comp['CALOBSVI'], color=ssn_data.Clr[4], linewidth=4, alpha=1)
        ax1.plot(pltx, Grp_Comp['SINGLETHreal'], 'g--', linewidth=2, alpha=1)

        ax1.set_xlim(left=np.min(ssn_data.ObsDat['FRACYEAR']) - 7, right=np.max(ssn_data.ObsDat['FRACYEAR']) + 7)
        ax1.set_ylim(bottom=0, top=maxplt * 1.25)
        ax1.set_ylabel('Average Number of Groups')

        ax1.legend(['Smoothed Ref. GN', 'Smoothed Ref. GN - Single Threshold', 'Smoothed Ref. GN - Multi-Threshold',
                    'Smoothed GN for ' + ssn_data.NamObs.capitalize()], 'Smoothed Ref. GN - Real Threshold',
                   loc='upper center', ncol=2, frameon=True, edgecolor='none', fontsize=18)

        MRE = np.round(np.nanmean(np.divide(Grp_Comp['MULTITH'] - Grp_Comp['CALOBSVI'], Grp_Comp['CALOBS'])),
                              decimals=2)
        MREav = np.round(np.nanmean(Grp_Comp['MULTITH'] - Grp_Comp['CALOBSVI']) / np.mean(Grp_Comp['CALOBS']),
                        decimals=2)
        slp = np.round(np.nanmean(Grp_Comp['MULTITH'] / Grp_Comp['CALOBSVI']), decimals=2)
        ax1.text(0.005, 0.2, 'MRE:' + str(MRE), horizontalalignment='left', verticalalignment='center',
                 transform=ax1.transAxes)
        ax1.text(0.005, 0.125, 'MNEav:' + str(MREav), horizontalalignment='left', verticalalignment='center',
                 transform=ax1.transAxes)
        ax1.text(0.005, 0.055, 'slp:' + str(slp), horizontalalignment='left', verticalalignment='center',
                 transform=ax1.transAxes)

        for Idx in range(0, ssn_data.cenPoints['OBS'].shape[0]):
            if ssn_data.vldIntr[Idx]:
                ax1.fill([ssn_data.endPoints['OBS'][Idx, 0], ssn_data.endPoints['OBS'][Idx, 0],
                          ssn_data.endPoints['OBS'][Idx + 1, 0], ssn_data.endPoints['OBS'][Idx + 1, 0]],
                         [0, maxplt * 1.25, maxplt * 1.25, 0], color=ssn_data.Clr[1 + np.mod(Idx, 2) * 2], alpha=0.2,
                         linestyle=None)
                ax1.text(ssn_data.cenPoints['OBS'][Idx, 0], maxplt * 0.05,
                         'Th:' + str(np.round(ssn_data.wAvI[Idx], decimals=1)),
                         horizontalalignment='center', verticalalignment='center')

    fig.savefig(figure_path, bbox_inches='tight')

    fig.clear()
    plt.close(fig)

    print('done.', flush=True)
    print(' ', flush=True)
