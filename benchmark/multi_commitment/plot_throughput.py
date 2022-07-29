#!/usr/bin/python3
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
import numpy as np

import csv

NR = [6, 5, 4, 3]
WORD_SIZES = [0, 1, 4, 8, 32]

def read_summary_file(file):
    with open(file, 'r') as fd:
        summary_rows = list(csv.reader(fd))
        summary = [[], [], [], []] # 10^0, 10^1, 10^2, 10^3 commitments

        c = 0
        counter = 0
        row_counter = 0

        for r in summary_rows:
            try:
                values_r = [float(v) for v in r[0].split('\t')]
                summary[c].append((values_r[0], values_r[1], values_r[2], values_r[7]))
                counter += 1

                if counter == NR[c]:
                    c = (c + 1) % 4
                    counter = 0
            except:
                pass

    return summary

base_dir = 'benchmark/multi_commitment/.results'
summary_pip_cpu = read_summary_file(base_dir + '/summary_pip-cpu.txt')
summary_naive_cpu = read_summary_file(base_dir + '/summary_naive-cpu.txt')

figs, ax = plt.subplots(2, 2)

figs.set_size_inches(18.5, 10.5, forward=True)

plt.subplots_adjust(
    # left=0.1,
    # bottom=0.1, 
    # right=0.9, 
    # top=0.9, 
    # wspace=0.4, 
    hspace=0.3
)

for c in range(len(NR)):
    colors = ['b', 'r', 'navy', 'y', 'black']
    symbols = ['v-', '*-', 'o-', 's-', '+-']

    def plot_summary(summary, summary_name, marker):
        minv, maxv = 1e20, 0
        for ws in range(len(WORD_SIZES)):
            num_commitments = [10**i for i in range(NR[c])]
            throughput = [summary[c][ws * NR[c] + i][3] for i in range(NR[c])]
            
            name = ''+str(WORD_SIZES[ws]) + ' bytes - ' + summary_name

            ax[c//2][c%2].plot(
                num_commitments,
                throughput,
                marker=marker,
                label=name,
                color=colors[ws]
            )

            for a, b in zip(num_commitments, throughput):
                minv = min(minv, b)
                maxv = max(maxv, b)
                ax[c//2][c%2].text(
                    a, b, f'{b:.1E}',
                    color=colors[ws],
                    fontsize=7,
                    rotation=35
                )

        return minv, maxv

    minv1, maxv1 = plot_summary(summary_pip_cpu, 'pip cpu', '*')
    minv2, maxv2 = plot_summary(summary_naive_cpu, 'naive cpu', 's')

    divider = make_axes_locatable(ax[c//2][c%2])
    ax[c//2][c%2].grid(color='w', linestyle='--')
    ax[c//2][c%2].set_xscale('log')
    ax[c//2][c%2].set_yscale('log')
    ax[c//2][c%2].set_ylim(min(minv1, minv2) / 1.7, 1.7 * max(maxv1, maxv2))

    ax[c//2][c%2].set_xlabel('Commitment Length')

    if c%2 == 0:
        ax[c//2][c%2].set_ylabel('Throughput \n(Exponentiations / second)')

    if c == 3:
        ax[c//2][c%2].legend(
            loc=8,
            ncol=2,
            handleheight=2.4,
            labelspacing=0.05,
            bbox_to_anchor=(1.4, 0.9)
        )

    ax[c//2][c%2].grid(visible=True, which='minor', linestyle='--')

    ax[c//2][c%2].set_title('Number of Commitments: ' + str(10**c))


plt.savefig(base_dir + '/commitments_pip_vs_naive_cpu.svg', bbox_inches='tight', pad_inches=0.1)
