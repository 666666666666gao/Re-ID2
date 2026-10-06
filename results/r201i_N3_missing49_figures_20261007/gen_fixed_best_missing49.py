"""All fixed-checkpoint query/gallery cells; disjoint private-coordinate limits marked."""
from matplotlib.patches import Patch, Rectangle
import numpy as np
from plot_style import DATA, plt, save

fig, axes = plt.subplots(2, 3, figsize=(7.6, 5.9))
subsets = DATA['subsets']
for row, metric in enumerate(('mAP', 'Rank-1')):
    maximum = max(abs(v) for d in DATA['matrices'].values() for line in d[metric] for v in line)
    for column, dataset in enumerate(('RGBNT201', 'RGBNT100', 'MSVR310')):
        ax = axes[row, column]
        matrix = np.asarray(DATA['matrices'][dataset][metric])
        image = ax.pcolormesh(np.arange(8)-.5, np.arange(8)-.5, matrix,
            cmap='RdBu_r', vmin=-maximum, vmax=maximum, shading='flat')
        ax.set_aspect('equal')
        ax.set_ylim(6.5, -.5)
        for q, query in enumerate(subsets):
            for g, gallery in enumerate(subsets):
                if not set(query).intersection(gallery):
                    ax.add_patch(Rectangle((g-.5, q-.5), 1, 1, facecolor='none', edgecolor='#777777', hatch='///', linewidth=.25))
                value = matrix[q, g]
                ax.text(g, q, f'{value:+.1f}', ha='center', va='center', fontsize=6.5,
                    color='white' if abs(value) > maximum*.65 else 'black')
        ax.set_xticks(range(7), subsets, rotation=45)
        ax.set_yticks(range(7), subsets)
        ax.set_xlabel('Gallery availability\n' + dataset)
        if column == 0:
            ax.set_ylabel('Query availability\n' + metric + ' change')
    color_axis = fig.add_axes([.903, .59 if row == 0 else .15, .015, .28])
    fig.colorbar(image, cax=color_axis, label='Percentage points')
fig.legend(handles=[Patch(facecolor='none', edgecolor='#777777', hatch='///',
    label='Disjoint sources: nearconstant private distance; tie-sensitive')],
    loc='upper center', bbox_to_anchor=(.5, .995), frameon=False)
fig.subplots_adjust(top=.88, bottom=.10, left=.10, right=.875, hspace=.45, wspace=.35)
save(fig, 'fixed_best_missing49')
