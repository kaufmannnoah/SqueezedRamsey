from matplotlib.gridspec import GridSpec
import matplotlib.pyplot as plt
from AAfunctions.fun_clean import *
from param import *

set_plot_styles()
c = colors()
c_light = ['#F2C866', '#66A9D9', '#E89A66', '#66C6A9']
c_light = ['#F6DDAA', '#AAD4E9', '#F2C2AA', '#AAE9D2']


ls = ['', '--', '']
marker = ['o', '', 'D']

# Apply the styles
fig_width = 3.375  # in inches
fig_height = fig_width # Aspect ratio for a visually pleasing layout
fig = plt.figure(figsize=(fig_width, fig_height))

# Define the grid layout
width_ratios = [1, 1]
height_ratios = [5, 3]
gs = GridSpec(2, 2, width_ratios=width_ratios, height_ratios=height_ratios, figure=fig)

ax2 = fig.add_subplot(gs[0, :])  # Spans both rows in column 1
ax3 = fig.add_subplot(gs[1, 0])  # Spans both rows in column 1
ax4_bottom = fig.add_subplot(gs[1, 1])  # Spans both rows in column 2

# Adjust layout with increased padding to add space between subplots
plt.tight_layout()

# Add axis labels and titles
ax2.set_xlabel(r'$N$', labelpad=-1)
ax3.set_xlabel(r'$N$', labelpad=-1)
ax4_bottom.set_xlabel(r'$N$', labelpad=-1)

ax2.set_ylabel(r'$\sqrt{N T} \Delta b_{\text{est}}$', labelpad=-3)
ax3.set_ylabel(r'$r$', labelpad=-1)
ax4_bottom.set_ylabel(r'$\tau_{\text{opt}}$', labelpad=-1)

# set scaling for x-axis
ax2.set_xscale('log')
ax3.set_xscale('log')
ax4_bottom.set_xscale('log')

#set limits for x-axis
ax2.set_xlim((1, 10**4))
ax3.set_xlim((1, 10**4))
ax4_bottom.set_xlim((1, 10**4))

#set scaling for y-axis
ax2.set_yscale('linear')
ax3.set_yscale('log')
ax4_bottom.set_yscale('log')

#set limits for y-axis
ax2.set_ylim((0, 2.13))
ax2.set_yticks([0, 1, np.sqrt(np.e)])
ax2.set_yticklabels(['0', '1', r'$\sqrt{e}$'])
ax3.set_ylim((0.7, 10))
ax3.set_yticks([1, 10])
ax3.set_yticklabels(['1', '10'])
ax4_bottom.set_yticks([0.0001, 0.01, 1])
ax4_bottom.set_yticklabels([r'$10^{\text{-}4}$', r'$10^{\text{-}2}$',  r'$10^{0}$'])

label_noise = ['Gaussian', 'White', 'Linear', 'Ohmic']

# Ploting
# Load data
step = 1
out = np.load(name_file)
for idn in range(4):
    TT = T[idn]
    normalization = np.sqrt(1 / (TT * N[::step]))
    w = np.logspace(-1, 1, 1001)
    tau = np.linspace(0, 10, 1001)
    ax2.plot(N[::step], np.sqrt(out[idn, ::step, 0]) / normalization, marker=marker[0], c=c_light[idn], ls=ls[0], zorder = 1, alpha=1)
    ax2.plot(N[::step], np.sqrt(out[idn, ::step, 3]) / normalization, marker=marker[1], c=c_light[idn], ls=ls[1], zorder= 0, alpha= 1)  
    ax2.plot(N[::step], np.sqrt(out[idn, ::step, 6]) / normalization, marker=marker[2], label= label_noise[idn], c=c[idn], ls=ls[2], zorder= 2, alpha= 1)     
    ax3.plot(N[::2*step], np.sqrt(out[idn, ::2*step, 3]) / np.sqrt(out[idn, ::2*step, 0]), label= label_noise[idn], marker= marker[0], ls=ls[0], c=c_light[idn], zorder=1, alpha=1)
    ax3.plot(N[::2*step], np.sqrt(out[idn, ::2*step, 3]) / np.sqrt(out[idn, ::2*step, 6]), marker=marker[2], ls=ls[2], c=c[idn], zorder= 2, alpha=1)
    ax4_bottom.plot(N[::2*step], out[idn, ::2*step, 7], marker=marker[2], c=c[idn], ls=ls[2], zorder= 10 - idn//2, alpha=1)

    if idn == 2:
        ax4_bottom.plot(N[::4 * step], out[idn, ::4 * step, 7], marker[2], c=c[idn], ls=ls[2], zorder= 10 + idn//2, alpha=1)
    if idn == 3:
        ax4_bottom.plot(N[::4 * step], out[idn, ::4 * step, 7], marker[2], c=c[idn], ls=ls[2], zorder= 10 + idn//2, alpha=1)

ax2.plot(N, np.ones(len(N)), color='gray', linestyle='-', zorder=-1, alpha=0.7, lw=0.5)
ax3.plot(N, np.ones(len(N)), color='gray', linestyle='-', zorder=-1, alpha=0.7, lw=0.5)


# Dummy plots for legend handles
ghz_handle, = ax2.plot(10, 10, marker=marker[2], c='gray', ls=ls[2], label='GHZ')
squeezed_handle, = ax2.plot(10, 10, marker=marker[0], c='gray', ls=ls[0], label='Squeezed')
separable_handle, = ax2.plot(10, 10, marker=marker[1], c='gray', ls=ls[1], label='Separable')

ax2.legend(frameon=True, handlelength=1.5, handletextpad=0.5, borderpad=0.5, borderaxespad=0.5)

# Second legend (bottom left, in background)
second_legend = ax2.legend(
    handles=[ghz_handle, squeezed_handle, separable_handle],
    loc='upper left',
    frameon=True,
    handlelength=1.7,
    handletextpad=0.2,
    borderpad=0.5,
    borderaxespad=0.5,
    ncol=3,
    columnspacing=1
)
ax2.add_artist(second_legend)
second_legend.set_zorder(0)  # Put legend in the background
second_legend.get_frame().set_linewidth(0.5)

handles_noise = []
for i in range(4):
    handles_noise = handles_noise + ax2.plot(10, 10, marker='D', c=c[i], ls=ls[2], label=label_noise[i])

first_legend = ax2.legend(
    handles=handles_noise,
    loc='lower left',
    frameon=True,
    handlelength=1,
    handletextpad=0.2,
    borderpad=0.5,
    borderaxespad=0.5,
    ncol=2,
    columnspacing=0
)
ax2.add_artist(first_legend)
first_legend.set_zorder(0)  # Put legend in the background
first_legend.get_frame().set_linewidth(0.5)


# Add bold letters (a)-(e) to the bottom right of each subplot
labels = ['a','b', 'c']
axes = [ax2, ax3, ax4_bottom]
for ax, label in zip(axes[:3], labels[:3]):
    if label == 'a':
        ax.text(0.99, 1 - 0.02/8*3, label, transform=ax.transAxes, ha='right', va='top', fontweight='bold', zorder=10)
    else:
        ax.text(0.98, 1 - 0.02/8*5, label, transform=ax.transAxes, ha='right', va='top', fontweight='bold', zorder=10)

plt.savefig('fig6.pdf', bbox_inches='tight')
plt.show()