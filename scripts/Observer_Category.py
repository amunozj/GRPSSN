import pandas as pd
import numpy as np
from SSN_Config import SSN_ADF_Config as config
from collections import defaultdict
import csv
import matplotlib
from collections import Counter

matplotlib.use('MacOSX')
import matplotlib.pyplot as plt


outfile = 'observer_categories.csv'

input_dir = '~/Desktop/Run-2018-6-8'
flag_files = {'AO': '{}/{}Observer_ADF.csv'.format(input_dir, config.get_file_prepend('ADF', 'OBS')),
              'AM': '{}/{}Observer_ADF.csv'.format(input_dir, config.get_file_prepend('ADF', 'FULLM')),
              'QO': '{}/{}Observer_ADF.csv'.format(input_dir, config.get_file_prepend('QDF', 'OBS')),
              'QM': '{}/{}Observer_ADF.csv'.format(input_dir, config.get_file_prepend('QDF', 'FULLM'))}

# Size definitions
dpi = 300
pxx = 3000  # Horizontal size of each panel
pxy = pxx  # Vertical size of each panel
frc = 1  # Fraction of the panel devoted to histograms

nph = 1  # Number of horizontal panels
npv = 1  # Number of vertical panels

# Padding
padv = 200  # Vertical padding in pixels
padv2 = 200  # Vertical padding in pixels between panels
padh = 200  # Horizontal padding in pixels at the edge of the figure
padh2 = 200  # Horizontal padding in pixels between panels

# Figure sizes in pixels
fszv = (npv * pxy + 2 * padv + (npv - 1) * padv2)  # Vertical size of figure in inches
fszh = (nph * pxx + 2 * padh + (nph - 1) * padh2)  # Horizontal size of figure in inches

# Conversion to relative unites
ppadv = padv / fszv  # Vertical padding in relative units
ppadv2 = padv2 / fszv  # Vertical padding in relative units
ppadh = padh / fszv  # Horizontal padding the edge of the figure in relative units
ppadh2 = padh2 / fszv  # Horizontal padding between panels in relative units


def make_best_category(outfile, flag_files, r2_threshold=0.05, use_NA=False):
    bases = ['R2']  # Only supports 1 base right now
    # fields = ['AvThreshold', 'Avg.Res', 'SDThreshold', 'R2OO', 'Avg.ResOO', 'R2DT', 'Avg.ResDT']
    fields = []

    header = ['Observer', 'Flag']
    for b in bases:
        header += [b]
        for f in fields:
            header += ['{}_{}'.format(b, f)]

    obs_cats = defaultdict(dict)

    for FLAG, path in flag_files.items():
        data = pd.read_csv(path, quotechar='"', encoding='utf-8', header=0)
        for r in fields + bases:
            data = data[np.isfinite(data[r])]

        for index, row in data.iterrows():
            obs_id = row['Observer']
            if obs_id in obs_cats.keys():
                for i in range(0, len(bases)):
                    r = bases[i]
                    cur_max_r2 = obs_cats[obs_id][r][1]
                    if row[r] > cur_max_r2 + r2_threshold:
                        fs = [row[r]] + [row[f] for f in fields]
                        obs_cats[obs_id][r] = [FLAG] + fs + [False]
                    elif row[r] == obs_cats[obs_id][r][1] or row[r] > cur_max_r2:
                        if use_NA:
                            obs_cats[obs_id][r][-1] = True
                        else:
                            obs_cats[obs_id][r][0] += '~{}'.format(FLAG)
            else:
                for r in bases:
                    fs = [row[r]] + [row[f] for f in fields]
                    obs_cats[obs_id][r] = [FLAG] + fs + [False]

    writer = csv.writer(open(outfile, 'w'), delimiter=' ')

    writer.writerow(header)

    int_flags = False
    flags_to_int = {'AO': 0, 'AM': 1, 'QO': 2, 'QM': 3}

    keys = sorted(obs_cats.keys())
    for key in keys:
        r_dict = obs_cats[key]

        cats = [v[0] if not v[-1] else 'NA' for v in r_dict.values()]

        if not use_NA and 'NA' in cats:
            continue

        if int_flags:
            cats = [flags_to_int[cats[0]]]

        # sum turns the double list into a single list
        nums = sum([[round(j, 3) for j in v[1:-1]] if not v[-1] else [0 for _ in v[1:-1]] for v in r_dict.values()], [])

        r = [key] + cats + nums

        writer.writerow(r)


def plot_best(file, vars=('R2', 'R2OO', 'R2DT'), show_plot=True):
    with open(file, 'r') as w:
        # Read csv file with flag data
        rows = []
        reader = csv.reader(w, delimiter=' ')
        for r in reader:
            rows.append(r)

        # Set up plot
        if show_plot:
            fig = plt.figure(figsize=(fszh / dpi, fszv / dpi))
            ax = fig.add_axes([ppadh, ppadv, pxx / fszh * frc, pxy / fszv * frc])
            # ax.set_title('Best flag combinations based on different calculations of R^2')

        Clr = [(0.00, 0.00, 0.00),
               (0.31, 0.24, 0.00),
               (0.43, 0.16, 0.49),
               (0.32, 0.70, 0.30),
               (0.45, 0.70, 0.90),
               (1.00, 0.82, 0.67)]

        flag_cols = {'AM': Clr[4], 'AO': Clr[3],
                     'QM': Clr[2], 'QO': Clr[5]}

        # Find indices of flag variables in csv file
        inds = {v: rows[0].index(v) - 1 for v in vars}

        # Set up main data storage
        data = []
        labels = []
        obs_dict = defaultdict(list)

        for flag, color in flag_cols.items():
            pts_dict = defaultdict(list)
            labels.append(flag)
            for row in rows[1:]:
                for r2_type, flag_index in inds.items():
                    if flag in row[flag_index]:
                        obs_dict[flag].append(row[0])
                        pts_dict[r2_type].append(float(row[flag_index + 1]))

            # Iterate of pts_dict to make sure data is in right order
            for r2_type in vars:
                pts = pts_dict[r2_type]
                data.append(pts)
            data.append([])

        data = data[:-1]
        labels.append('')
        dlengths = [len(d) for d in data]

        upperLabels = (list(vars) + ['']) * len(flag_cols.keys())

        if show_plot:
            ax.set_ylim(0, 1)
            bot, top = ax.get_ylim()
            numBoxes = len(data)
            pos = np.arange(numBoxes) + 1
            for tick in range(numBoxes):
                ax.text(pos[tick], top - (top * 0.03), upperLabels[tick],
                        horizontalalignment='center', size='small')
                ax.text(pos[tick], top - (top * 0.07),
                        ('{} pts'.format(dlengths[tick]) if dlengths[tick] is not 0 else ''),
                        horizontalalignment='center', size='x-small')

            bplot = ax.boxplot(data, patch_artist=True)

            # box = plt.boxplot(data, notch=True, patch_artist=True)

            colors = ['red', 'lightgreen', 'yellow', (0, 0, 0)] * len(flag_cols.keys())
            for patch, color in zip(bplot['boxes'], colors):
                patch.set_facecolor(color)

        labs = []
        for l in labels:
            labs += ['', l] + [''] * (len(vars) - 1)

        if show_plot:
            ax.set_xticklabels(labs, rotation=45, fontsize=12)
            plt.show()

        return labels, data, obs_dict


def plot_all(flag_files, make_cat_file=True, use_NA=False, r2_threshold=0.05, var='R2'):

    if make_cat_file:
        make_best_category(outfile, flag_files, use_NA=use_NA, r2_threshold=r2_threshold)

    all_data = {}

    for flags, path in flag_files.items():
        data = pd.read_csv(path, quotechar='"', encoding='utf-8', header=0)
        data = data[np.isfinite(data[var])]
        all_data[flags] = data

    fig = plt.figure(figsize=(fszh / dpi, fszv / dpi))
    ax = fig.add_axes([ppadh, ppadv, pxx / fszh * frc, pxy / fszv * frc])


    alpha = 0.3
    Clr = [(0.00, 0.00, 0.00, alpha),
           (0.31, 0.24, 0.00, alpha),
           (0.43, 0.16, 0.49, alpha),
           (0.32, 0.70, 0.30, alpha),
           (0.45, 0.70, 0.90, alpha),
           (1.00, 0.82, 0.67, alpha)]

    flag_cols = {'AM': Clr[4], 'AO': Clr[3],
                 'QM': Clr[2], 'QO': Clr[5]}

    labs, data, obs_dict = plot_best(outfile, [var], show_plot=False)
    labs = labs[:-1]

    all_data = []
    new_labs = []
    for i, f1 in enumerate(labs):
        obs_list_f1 = obs_dict[f1]
        flag_data = {}
        for j, f2 in enumerate(labs):
            d = {}
            ff = flag_files[f2]
            df = pd.read_csv(ff, quotechar='"', encoding='utf-8', header=0)
            for index, row in df.iterrows():
                obs_id = int(row['Observer'])
                if str(obs_id) in obs_list_f1 and np.isfinite(row[var]):
                    d[obs_id] = row[var]
                    continue
            flag_data[f2] = d
        dat = [ob for fd in flag_data.values() for ob in fd.keys()]
        cnt = Counter(dat)
        good_obs = [k for k, v in cnt.items() if v == len(flag_data.keys())]
        for k, f3 in enumerate(labs):
            good_data = [r for o, r in flag_data[f3].items() if o in good_obs]
            all_data.append(good_data)
            new_labs.append(f3)
        all_data.append([])
        new_labs.append('')
    all_data = all_data[:-1]
    new_labs = new_labs[:-1]

    ax.set_ylabel(var)
    ax.set_ylim(0, 1)
    bot, top = ax.get_ylim()
    numBoxes = len(all_data)
    pos = np.arange(numBoxes) + 1

    dlengths = [len(d) for d in all_data]
    print(dlengths)

    manual_index = [0, 6, 12, 18]

    cols = [flag_cols[l] if l is not '' else (0, 0, 0, 0) for l in new_labs]
    n = len(flag_cols.keys())
    for i, f in enumerate(new_labs[:4]):
        ax.text(pos[i * (n + 1) + 1] + 0.5, top - (top * 0.03), 'Observers best fit with {}'.format(f),
                horizontalalignment='center', size='small')
        ax.text(pos[i * (n + 1) + 1] + 0.5, top - (top * 0.07),
                ('{} pts'.format(dlengths[i * (n + 1) + 1]) if dlengths[i * (n + 1) + 1] is not 0 else ''),
                horizontalalignment='center', size='x-small')

    bplot = ax.boxplot(all_data, patch_artist=True)
    ax.set_xticklabels(new_labs, rotation=45, fontsize=12)

    for i, (patch, color) in enumerate(zip(bplot['boxes'], cols)):
        if i in manual_index:
            color = list(color)[:-1] + [1.0]
        patch.set_facecolor(color)

    if use_NA:
        ax.set_title('Observers with highlighted flag providing best fit, exclusive flag fit overlap')
    else:
        #ax.set_title('Observers with highlighted flag providing best fit, {}% inclusive flag fit overlap'.format(r2_threshold*100))
        ax.set_title('Performance of ADF calculation methods')

    plt.show()


plot_all(flag_files, True, use_NA=False, r2_threshold=0.05, var='R2')
