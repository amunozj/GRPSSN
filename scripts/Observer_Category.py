import pandas as pd
import numpy as np
from SSN_Config import SSN_ADF_Config as config
from collections import defaultdict
import csv
import matplotlib

matplotlib.use('MacOSX')
import matplotlib.pyplot as plt


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


def make_best_category(outfile, flag_files):


    #row_names = ['Observer', 'R2', 'R2', 'R2OO', 'R2OO', 'R2DT', 'R2DT']
    #row_names = ['Observer', 'R2', 'R2', 'AvThreshold', 'AvThreshold', 'Avg.Res', 'Avg.Res']
    bases = ['R2'] # Only supports 1 base right now
    fields = ['AvThreshold', 'Avg.Res', 'SDThreshold', 'R2OO', 'Avg.ResOO', 'R2DT', 'Avg.ResDT']

    header = ['Observer', 'Flag']
    for b in bases:
        header += [b]
        for f in fields:
            header += ['{}_{}'.format(b,f)]


    obs_cats = defaultdict(dict)

    for flags, path in flag_files.items():
        data = pd.read_csv(path, quotechar='"', encoding='utf-8', header=0)
        for r in fields + bases:
            data = data[np.isfinite(data[r])]

        for index, row in data.iterrows():
            if row['Observer'] in obs_cats.keys():
                for i in range(0, len(bases)):
                    r = bases[i]
                    if row[r] > obs_cats[row['Observer']][r][1]:
                        fs = [row[r]] + [row[f] for f in fields]
                        print(fs)
                        obs_cats[row['Observer']][r] = [flags] + fs + [False]
                    elif row[r] == obs_cats[row['Observer']][r][1]:
                        obs_cats[row['Observer']][r][-1] = True
            else:
                for r in bases:
                    fs = [row[r]] + [row[f] for f in fields]
                    obs_cats[row['Observer']][r] = [flags] + fs + [False]

    writer = csv.writer(open(outfile, 'w'), delimiter=' ')

    writer.writerow(header)

    flags_to_int = {'AO': 0, 'AM': 1, 'QO': 2, 'QM': 3}

    keys = sorted(obs_cats.keys())
    for key in keys:
        r_dict = obs_cats[key]

        cats = [v[0] if not v[-1] else 'NA' for v in r_dict.values()]

        if 'NA' in cats:
            continue

        cats = [flags_to_int[cats[0]]]

        # sum turns the double list into a single list
        nums = sum([[round(j, 3) for j in v[1:-1]] if not v[-1] else [0 for _ in v[1:-1]] for v in r_dict.values()], [])


        r = [key] + cats + nums
        # r = [key] + [x for z in zip(cats, nums) for x in z]




        writer.writerow(r)


outfile = 'observer_categories_new.csv'


input_dir = '~/Desktop/Run-2018-6-8'
flag_files = {'AO': '{}/{}Observer_ADF.csv'.format(input_dir, config.get_file_prepend('ADF', 'OBS')),
              'AM': '{}/{}Observer_ADF.csv'.format(input_dir, config.get_file_prepend('ADF', 'FULLM')),
              'QO': '{}/{}Observer_ADF.csv'.format(input_dir, config.get_file_prepend('QDF', 'OBS')),
              'QM': '{}/{}Observer_ADF.csv'.format(input_dir, config.get_file_prepend('QDF', 'FULLM'))}

make_best_category(outfile, flag_files)


def plot_best(file, vars=('R2', 'R2OO', 'R2DT'), show_plot=True):
    with open(file, 'r') as w:

        rows = []
        reader = csv.reader(w, delimiter=' ')
        for r in reader:
            rows.append(r)

        if show_plot:
            fig = plt.figure(figsize=(fszh / dpi, fszv / dpi))
            ax = fig.add_axes([ppadh, ppadv, pxx / fszh * frc, pxy / fszv * frc])
        #ax.set_title('Best flag combinations based on different calculations of R^2')

        Clr = [(0.00, 0.00, 0.00),
               (0.31, 0.24, 0.00),
               (0.43, 0.16, 0.49),
               (0.32, 0.70, 0.30),
               (0.45, 0.70, 0.90),
               (1.00, 0.82, 0.67)]

        flag_cols = {'AM': Clr[4], 'AO': Clr[3],
                     'QM': Clr[2], 'QO': Clr[5]}

        inds = {v: rows[0].index(v) for v in vars}

        data = []
        labels = []
        obs_dict = defaultdict(list)
        for flag, color in flag_cols.items():
            pts_dict = defaultdict(list)
            labels.append(flag)
            for row in rows[1:]:
                for r2, i in inds.items():
                    if row[i] == flag:
                        obs_dict[flag].append(row[0])
                        pts_dict[r2].append(float(row[i + 1]))

            for r2 in vars:
                pts = pts_dict[r2]
                data.append(pts)
            data.append([])

        data = data[:-1]
        labels.append('')

        dlengths = [len(d) for d in data]

        upperLabels = (list(vars) + ['']) * len(flag_cols.keys())

        #upperLabels = ['{} ({})'.format(l, d) if d is not 0 else '' for l, d in zip(upperLabels, dlengths)]


        if show_plot:
            ax.set_ylim(0, 1)
            bot, top = ax.get_ylim()
            numBoxes = len(data)
            pos = np.arange(numBoxes) + 1
            for tick in range(numBoxes):
                ax.text(pos[tick], top - (top * 0.03), upperLabels[tick],
                        horizontalalignment='center', size='small')
                ax.text(pos[tick], top - (top * 0.07), ('{} pts'.format(dlengths[tick]) if dlengths[tick] is not 0 else ''),
                        horizontalalignment='center', size='x-small')

            bplot = ax.boxplot(data, patch_artist=True)

            # box = plt.boxplot(data, notch=True, patch_artist=True)

            colors = ['red', 'lightgreen', 'yellow', (0,0,0)] * len(flag_cols.keys())
            for patch, color in zip(bplot['boxes'], colors):
                patch.set_facecolor(color)

        labs = []
        for l in labels:
            labs += ['', l] + ['']*(len(vars)-1)

        if show_plot:
            ax.set_xticklabels(labs, rotation=45, fontsize=12)
            plt.show()

        return labels, data, obs_dict


#plot_best(outfile, vars=['R2', 'R2OO', 'R2DT'])


def plot_all(flag_files, var = 'R2'):

    all_data = {}

    for flags, path in flag_files.items():
        data = pd.read_csv(path, quotechar='"', encoding='utf-8', header=0)
        data = data[np.isfinite(data[var])]
        all_data[flags] = data

    fig = plt.figure(figsize=(fszh / dpi, fszv / dpi))
    ax = fig.add_axes([ppadh, ppadv, pxx / fszh * frc, pxy / fszv * frc])
    #ax.set_title('Best flag combinations based on different calculations of R^2')

    Clr = [(0.00, 0.00, 0.00, 0.5),
           (0.31, 0.24, 0.00, 0.5),
           (0.43, 0.16, 0.49, 0.5),
           (0.32, 0.70, 0.30, 0.5),
           (0.45, 0.70, 0.90, 0.5),
           (1.00, 0.82, 0.67, 0.5)]

    flag_cols = {'AM': Clr[4], 'AO': Clr[3],
                 'QM': Clr[2], 'QO': Clr[5]}


    labs, data, obs_dict = plot_best(outfile, [var], show_plot=False)
    labs = labs[:-1]
    data = [d for d in data if d]
    #print(labs, data, obs_dict)

    all_data = []
    new_labs = []
    for i, l in enumerate(labs):

        for j, flag in enumerate(labs):
            f = flag_files[flag]
            d = []
            df = pd.read_csv(f, quotechar='"', encoding='utf-8', header=0)
            for ob in obs_dict[l]:

                for index, row in df.iterrows():
                    if int(row['Observer']) == int(ob) and np.isfinite(row[var]):
                        d.append(row[var])
                        continue

            all_data.append(d)
            new_labs.append(flag)
        all_data.append([])
        new_labs.append('')

    all_data = all_data[:-1]
    new_labs = new_labs[:-1]

    ax.set_ylim(0, 1)
    bot, top = ax.get_ylim()
    numBoxes = len(all_data)
    pos = np.arange(numBoxes) + 1
    print(pos)
    dlen = [len(d) for d in all_data]



    cols = [flag_cols[l] if l is not '' else (0,0,0,0) for l in new_labs]

    n = len(flag_cols.keys())
    for i, f in enumerate(new_labs[:4]):
        print(pos[i*(n+1)+1])
        ax.text(pos[i*(n+1)+1] + 0.5, top - (top * 0.03), 'Observers best fit with {}'.format(f),
               horizontalalignment='center', size='small')
        #ax.text(pos[tick], top - (top * 0.07), ('{} pts'.format(dlengths[tick]) if dlengths[tick] is not 0 else ''),
         #      horizontalalignment='center', size='x-small')5

    bplot = ax.boxplot(all_data, patch_artist=True)
    ax.set_xticklabels(new_labs, rotation=45, fontsize=12)

    # box = plt.boxplot(data, notch=True, patch_artist=True)

    #colors = ['red', 'lightgreen', 'yellow', (0,0,0)] * len(flag_cols.keys())

    manual_index = [0, 6, 12, 18]

    for i, (patch, color) in enumerate(zip(bplot['boxes'], cols)):
        if i in manual_index:
            color = list(color)[:-1] + [1.0]
        patch.set_facecolor(color)

    plt.show()

#plot_all(flag_files, 'R2')

