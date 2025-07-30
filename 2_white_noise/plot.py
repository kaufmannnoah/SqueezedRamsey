from matplotlib.gridspec import GridSpec
import matplotlib.pyplot as plt
from AAfunctions.fun_clean import *
from param import *


# Define a function to set consistent plot styles
set_plot_styles()
c = colors()

# Define figure dimensions
fig_width = 3.375  # in inches
fig_height = 3 * fig_width / 5
fig = plt.figure(figsize=(fig_width, fig_height))

# Define the grid layout
width_ratios = [1, 1]
height_ratios = [1, 1]
gs = GridSpec(2, 2, width_ratios=width_ratios, height_ratios=height_ratios, figure=fig)
axN = fig.add_subplot(gs[:, 1])  # Spans both rows in column 1
axT = fig.add_subplot(gs[:, 0])  # Spans both rows in column 2
plt.tight_layout()

#set limits
axN.set_xlabel(r'$N$', labelpad=0)
axN.set_ylabel(r'$r$', labelpad=-3)
axN.set_ylim((1, np.sqrt(np.e)+0.02))
axN.set_yticks([1, 1.5, np.sqrt(np.e)])
axN.set_yticklabels(['1', '1.5', r'$\sqrt{e}$'])
axN.set_xscale('log')
axN.set_xlim((1, 10**8))
axN.set_xticks([1, 10**4, 10**8])

axT.set_xlabel(r'$A T$', labelpad=0)
axT.set_ylabel(r'$\Delta b_{\text{est}}$', labelpad=-3)
axT.set_yscale('log')
axT.set_ylim((0.0001, 0.06))
axT.set_yticks([0.001, 0.01])
axT.set_yticklabels(['.001', '.01'])
axT.set_xscale('log')
axT.set_xlim((10, 10**5))
axT.set_xticks([10, 1000, 100000])

# Ploting
# Load data
out_N = np.load(name_file_N)
axN.plot(N_1, np.sqrt(out_N[:, 3]) / np.sqrt(out_N[:, 0]), c=c[1], lw= 1.5, zorder= 1, ls='-')
axN.axhline(np.sqrt(np.e), c='gray', lw= 0.5, zorder= 0)

out_T = np.load(name_file_T)
axT.plot(T_2, np.sqrt(out_T[:, 0]), c=c[1], zorder= -1, ls='-', label= 'Squeezed')
axT.plot(T_2, np.sqrt(out_T[:, 3]), c=c[1], zorder= -1, ls=':', label= 'Separable')
axT.plot(T_2, np.sqrt(1 / (T_2 * N_2)), c=c[1], zorder= -1, ls='--', label= 'SQL')
leg = axT.legend(handlelength=1.4)
leg.get_frame().set_linewidth(0.5)

labels = ['b','a'] # Plot Labels
axes = [axN, axT]
for ax, label in zip(axes, labels):
    ax.text(0.98, 0.01, label, transform=ax.transAxes, ha='right', va='bottom', fontweight='bold', zorder=10)

# Show the plot
plt.savefig('fig2.pdf', bbox_inches='tight')
plt.show()
