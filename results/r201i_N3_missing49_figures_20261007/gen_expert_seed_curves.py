"""All300 observed epochs, with same-normal-mAP checkpoint markers on every row."""
from matplotlib.lines import Line2D
from plot_style import DATA, plt, save

fig, axes = plt.subplots(3, 3, figsize=(7.4, 6.8), sharex=True, sharey='row')
styles = {'frequency_shared': dict(color='#D55E00', linestyle='--'),
          'axis_shared': dict(color='#0072B2', linestyle='-')}
for column, seed in enumerate((42, 43, 44)):
    for run in (x for x in DATA['runs'] if x['seed'] == seed):
        series = run['epochs']
        selected = series[run['selected_epoch'] - 1]
        for row, (metric, label) in enumerate((('mAP', 'mAP (%)'), ('Rank1', 'Rank-1 (%)'), ('loss', 'Training objective'))):
            ax = axes[row, column]
            style = styles[run['model']]
            ax.plot([x['epoch'] for x in series], [x[metric] for x in series], linewidth=1.4, **style)
            ax.scatter(selected['epoch'], selected[metric], s=18, color=style['color'], zorder=4)
            ax.scatter(50, series[-1][metric], s=22, marker='s', facecolors='none', edgecolors=style['color'], zorder=4)
            if metric in DATA['reference'] and run['model'] == 'axis_shared':
                ax.axhline(DATA['reference'][metric], color='#777777', linestyle=':', linewidth=.8)
            if column == 0:
                ax.set_ylabel(label)
            ax.set_xlim(0, 51)
            ax.set_xticks((1, 25, 50))
            ax.grid(axis='y', linewidth=.4, alpha=.3)
    axes[2, column].set_xlabel('Additional epoch (seed ' + str(seed) + ')')
handles = [Line2D([0], [0], label='Ordinary', **styles['frequency_shared']),
    Line2D([0], [0], label='Dual axis', **styles['axis_shared']),
    Line2D([0], [0], color='#777777', linestyle=':', label='Teacher50 reference'),
    Line2D([0], [0], color='black', marker='o', linestyle='', label='mAP-selected'),
    Line2D([0], [0], color='black', marker='s', markerfacecolor='none', linestyle='', label='Epoch50')]
fig.legend(handles=handles, loc='upper center', bbox_to_anchor=(.5, .995), ncol=3, frameon=False)
fig.subplots_adjust(top=.88, bottom=.09, left=.095, right=.985, hspace=.30, wspace=.12)
save(fig, 'expert_seed_curves')
