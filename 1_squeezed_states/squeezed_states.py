import numpy as np
import matplotlib.pyplot as plt
from fun_clean import *

set_plot_styles()
colors = ['#613169', '#397F6D']

# Parameters
NN = [1000, 1001]
kk = np.logspace(-2., 0, num=801)

# Labels and colors
ls = ['-', ':']
lw = [1.5, 2]
alpha = 0.5

# Create figure and subplots
fig, axs = plt.subplots(1, 1, figsize=(2.1, 1.9))

axs.set_xlim((kk[0], kk[-1]))
axs.set_yscale('log')
axs.set_xscale('log')
axs.set_ylim((1e-5, 2))
axs.set_yticks(np.logspace(-4, 0, num= 3))
axs.set_xlabel(r'$\kappa$', labelpad=-1)
axs.set_yticks([0.0001, 0.001, 0.01, 1])
axs.set_yticklabels([r'$10^{-4}$', r'$1/N$', r'$10^{-2}$', r'$10^0$'])

# Top subplot
for idn, N in enumerate(NN):
    out = np.ones((len(kk), 6))
    for idk, ki in enumerate(kk):
        out[idk, :3] = first_moments(N, ki)
        out[idk, 3:] = second_moments(N, ki)
    axs.plot(kk, out[:, 2]**2 * 4 / N**2, c=colors[0], ls= ls[idn], lw= lw[idn], zorder= 2 - idn)
    axs.plot(kk, out[:, 4] * 4 / N, c=colors[1], ls= ls[idn], lw= lw[idn], zorder= 2 - idn)
    axs.plot(kk, out[:, 2]**2 * 4 / N**2, c=colors[0], ls= ls[idn], lw= lw[idn], zorder= 6 - idn, alpha= alpha)
    axs.plot(kk, out[:, 4] * 4 / N, c=colors[1], ls= ls[idn], lw= lw[idn], zorder= 6 - idn, alpha= alpha)

# First legend for the colored lines
legend1 = axs.legend(
    handles=[
        axs.plot([], [], c=colors[1], lw=1.5, label=r'$4 / N \, \langle J^2_y \rangle$')[0],
        axs.plot([], [], c=colors[0], lw=1.5, label=r'$4 / N^2 \, \langle J_z \rangle^2$')[0]
    ],
    loc='lower right',
    frameon=True,
    framealpha=1,
)
legend1.get_frame().set_linewidth(0.5)
axs.add_artist(legend1)
axs.text(0.02, 0.98, 'c', transform=axs.transAxes, ha='left', va='top', fontweight='bold', zorder=10)

# Remove minor ticks
axs.minorticks_off()
axs.set_facecolor('white')
fig.patch.set_facecolor('none')

# Tight layout for better spacing
fig.tight_layout()

# Save and show plot
plt.savefig('fig1.svg', bbox_inches='tight')
plt.show()