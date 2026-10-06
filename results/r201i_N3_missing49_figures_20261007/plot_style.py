"""CPU-only figure style; data are sealed CSV reductions, never neural outputs."""
import json
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parent
DATA = json.loads((ROOT / 'inputs.json').read_text())
plt.rcParams.update({'font.family': 'DejaVu Serif', 'font.size': 9,
    'axes.labelsize': 9, 'xtick.labelsize': 8, 'ytick.labelsize': 8,
    'legend.fontsize': 8, 'axes.spines.top': False, 'axes.spines.right': False,
    'pdf.fonttype': 42, 'svg.fonttype': 'none'})

def save(fig, name):
    for suffix in ('pdf', 'svg', 'png'):
        fig.savefig(ROOT / (name + '.' + suffix), dpi=200)
    plt.close(fig)
