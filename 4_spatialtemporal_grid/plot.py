import numpy as np
import matplotlib.pyplot as plt
from AAfunctions.fun_clean import *
from param import *

# Set font sizes for A4 plot
set_plot_styles()
c = colors()
fc = ['#F9F2FA',  # very light lavender
      '#F1FBFC',  # very pale aqua
      '#F2FBF3']  # very soft mint green

#out = np.load(name_file)
out = np.load('data_2.npy')

fig, axs = plt.subplots(5, 5, figsize=(7, 3.4), gridspec_kw={'wspace': 0.4, 'hspace': 0.3})
fig.delaxes(axs[0, 0]) # Remove the top-left plot
for ax in axs.flat:
    ax.set_xscale('log')

lw = 1.5
ls = '-'

## SPATIAL LEGEND ##
k = np.logspace(-1, 1, 1001)
for s in range(1, 5):
    ax = axs[s, 0]
    ax.set_xlim((0.1, 10))
    ax.set_xticks([0.1, 1, 10])
    #ax.set_xticklabels(['.1', '1', '10'])
    ax.set_ylim((-0.4, 4.4))
    ax.set_yticks([0, 2, 4])
    ax.set_yticklabels([0, '', 4])
    ax.set_ylabel(r'$G(k)$', labelpad=-4)
    ax.plot(k, spat_spec[s-1](k), c='gray', lw=lw, ls=ls, alpha= 1, zorder= 0)
axs[4, 0].set_xlabel(r'$k$', labelpad=0)

## TEMPORAL PLOTS ##
w = np.logspace(-2, 2, 1001)
for t in range(1, 5):
    ax = axs[0, t]
    ax.xaxis.set_label_position('top')
    ax.xaxis.tick_top()
    ax.set_xlabel(r'$\omega$', labelpad=0)
    ax.set_xlim((0.1, 10))
    ax.set_xticks([0.1, 1, 10])
    ax.set_ylim((-0.4, 4.4))
    ax.set_yticks([0, 2, 4])
    ax.set_yticklabels([0, '', 4])
    ax.set_ylabel(r'$S(\omega)$', labelpad=-4)
    ax.plot(w, temp_spec[t-1](w), c=c[t-1], lw=lw, ls=ls, zorder = 0, alpha= 1)

## PLOTS ##
for i in range(1, 5):
    axs[4, i].set_xlabel(r'$N$', labelpad=0)

for i in range(1, 4):
    for j in range(0, 5):
        axs[i, j].set_xlabel(None)
        axs[i, j].set_xticklabels([])

for i in range(1, 5):
    for j in range(1, 5):
        axs[i, j].set_xlim((1, 10**4))
        axs[i, j].set_xticks([1, 10**2, 10**4])
        axs[i, j].set_ylabel(r'$r$', labelpad=-10)
        axs[i, j].set_yscale('log')
        axs[i, j].set_ylim((1, 10.001))
        axs[i, j].set_yticks([1, 10])
        axs[i, j].set_yticklabels(['1', '10'])
        axs[i, j].plot(N, np.sqrt(out[i-1, j-1, :, 3] / out[i-1, j-1, :, 0]) , marker= 'none', c= c[j-1], ls=ls, lw= lw, alpha= 1, zorder= 0)
        if j == 1 or j == 4: axs[i, j].set_facecolor(fc[1])
        else: axs[i, j].set_facecolor(fc[0])

for ax in [axs[1, 2], axs[2, 1], axs[1, 1]]:
    ax.set_yscale('linear')
    ax.set_ylim(1, 1.26)
    ax.set_yticks([1, 1.1, 1.2])
    ax.set_ylabel(r'$r$', labelpad=-10)
    ax.set_yticklabels([1, '', 1.2])
    ax.set_facecolor(fc[2])

axs[2, 2].set_yscale('linear')
axs[2, 2].set_ylim(1, 1.7)
axs[2, 2].set_yticks([1, 1.3, 1.6])
axs[2, 2].set_yticklabels([1, '', 1.6,])
axs[2, 2].set_ylabel(r'$r$', labelpad=-10)
axs[2, 2].set_facecolor(fc[2])


fig.add_artist(plt.Line2D([0.09, 0.9], [0.738, 0.738], color='gray', lw=1, transform=fig.transFigure, clip_on=False))
fig.add_artist(plt.Line2D([0.25, 0.25], [0.11, 0.96], color='gray', lw=1, transform=fig.transFigure, clip_on=False))
fig.text(0.248, 0.96, 'temporal', c='gray', va='top', ha='right', transform=fig.transFigure, rotation=90)
fig.text(0.09, 0.743, 'spatial', c='gray', va='bottom', ha='left', transform=fig.transFigure)

plt.tight_layout()
plt.savefig('fig4.svg', bbox_inches='tight')

plt.show()