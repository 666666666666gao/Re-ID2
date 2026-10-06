"""Deterministic CPU Matplotlib plots of sealed completed normal data; no Torch import."""
import json
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np

ROOT=Path(__file__).resolve().parent
DATA=json.loads((ROOT/'inputs.json').read_text())
MODELS=('G_axis_shared','I_axis_shared','G_frequency_shared','I_frequency_shared')
COLORS={'G_axis_shared':'#0072B2','I_axis_shared':'#0072B2','G_frequency_shared':'#D55E00','I_frequency_shared':'#D55E00'}
LABELS={'G_axis_shared':'G dual axis','I_axis_shared':'I dual axis','G_frequency_shared':'G ordinary','I_frequency_shared':'I ordinary'}
plt.rcParams.update({'font.family':'DejaVu Serif','font.size':10,'axes.labelsize':10,'xtick.labelsize':9,'ytick.labelsize':9,'legend.fontsize':9,'axes.spines.top':False,'axes.spines.right':False,'pdf.fonttype':42,'svg.fonttype':'none'})


def style(model):
    return dict(color=COLORS[model],linestyle='--' if model.startswith('G_') else '-',linewidth=1.6)


def save(fig,name):
    for extension in ('pdf','svg','png'):
        fig.savefig(ROOT/(name+'.'+extension),dpi=150)
    plt.close(fig)


def normal(dataset):
    group=DATA['datasets'][dataset]
    fig,axes=plt.subplots(2,3,figsize=(11.5,7))
    for ax,metric in zip(axes.flat,DATA['metrics']):
        for model in MODELS:
            rows=group[model]['epochs'];ax.plot([r['epoch'] for r in rows],[r[metric] for r in rows],label=LABELS[model],**style(model))
            selected=group[model]['selected_epoch'];ax.scatter(selected,rows[selected-1][metric],s=20,marker='o' if model.startswith('I_') else 's',color=COLORS[model],zorder=4)
        ax.axhline(group['original_DeMo']['selected_metrics'][metric],color='#777777',linestyle=':',linewidth=1.3)
        ax.set_xlabel('Expert-stage epoch');ax.set_ylabel(metric+' (%)');ax.set_xlim(0,51);ax.set_xticks([1,10,20,30,40,50]);ax.grid(axis='y',color='#dddddd',linewidth=.5)
    handles=[Line2D([0],[0],label=LABELS[m],**style(m)) for m in MODELS]+[Line2D([0],[0],color='#777777',linestyle=':',label='DeMo50 selected reference')]
    fig.legend(handles=handles,loc='upper center',bbox_to_anchor=(.5,.995),ncol=5,frameon=False)
    fig.subplots_adjust(top=.89,bottom=.11,left=.075,right=.99,hspace=.36,wspace=.30)
    save(fig,dataset+'_normal_six_curves')


def objectives():
    fig,axes=plt.subplots(2,3,figsize=(11.5,7))
    for row,dataset in enumerate(('RGBNT201','MSVR310')):
        group=DATA['datasets'][dataset]
        for model in MODELS:
            e=group[model]['epochs'];axes[row,0].plot([r['epoch'] for r in e],[r['loss'] for r in e],**style(model))
        for model in ('I_axis_shared','I_frequency_shared'):
            e=group[model]['primary'];axes[row,1].plot([r['epoch'] for r in e],[r['loss'] for r in e],**style(model))
            axes[row,2].plot([r['epoch'] for r in e],[r['violations'] for r in e],**style(model))
        for column,label in enumerate(('Total recorded loss','I primary hinge loss','I mean violations / 64 anchors')):
            ax=axes[row,column];ax.set_xlabel('Expert-stage epoch');ax.set_ylabel(dataset+'\n'+label);ax.set_xlim(0,51);ax.set_xticks([1,10,20,30,40,50]);ax.grid(axis='y',color='#dddddd',linewidth=.5)
    handles=[Line2D([0],[0],label=LABELS[m],**style(m)) for m in MODELS]
    fig.legend(handles=handles,loc='upper center',bbox_to_anchor=(.5,.995),ncol=4,frameon=False)
    fig.subplots_adjust(top=.89,bottom=.11,left=.08,right=.99,hspace=.36,wspace=.48)
    save(fig,'completed_normal_objectives')


def routes():
    fig,axes=plt.subplots(2,2,figsize=(8.3,7))
    for row,dataset in enumerate(('RGBNT201','MSVR310')):
        for column,model in enumerate(('G_axis_shared','I_axis_shared')):
            ax=axes[row,column];v=100*np.asarray(DATA['datasets'][dataset][model]['mean_route'])
            image=ax.imshow(v,vmin=0,vmax=40,cmap='Blues',aspect='auto')
            for y in range(7):
                for x in range(3):ax.text(x,y,f'{v[y,x]:.2f}',ha='center',va='center',fontsize=9,color='white' if v[y,x]>20 else 'black')
            ax.set_xticks([0,1,2],['Low','Middle','High']);ax.set_yticks(range(7),['R','N','T','RN','RT','NT','RNT'])
            ax.set_xlabel(('G' if column==0 else 'I')+' spatial-frequency band');ax.set_ylabel(dataset+' modality relation')
    fig.subplots_adjust(left=.10,right=.85,top=.98,bottom=.10,wspace=.38,hspace=.34)
    cax=fig.add_axes([.89,.15,.023,.73]);fig.colorbar(image,cax=cax,label='Mean route allocation (%)')
    save(fig,'selected_axis_route_means')
