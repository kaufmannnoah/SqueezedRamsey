from matplotlib.gridspec import GridSpec
import matplotlib.pyplot as plt
from AAfunctions.fun_clean import *
from param import *


# Define a function to set consistent plot styles
set_plot_styles()
c = colors()

ls = ['', '--']
m = ['o', '']
alpha = [1, 0.5]

# Define figure dimensions (7 inches wide for PRX Quantum two-column layout)
fig_width = 7  # in inches
fig_height = fig_width * (1 / 3)

# Create the figure
fig = plt.figure(figsize=(fig_width, fig_height))

# Define the grid layout
width_ratios = [6, 6, 3, 3]
height_ratios = [1, 1]
gs = GridSpec(2, 4, width_ratios=width_ratios, height_ratios=height_ratios, figure=fig)

# Create the subplots
ax2 = fig.add_subplot(gs[:, 0])  # Spans both rows in column 1
ax3 = fig.add_subplot(gs[:, 1])  # Spans both rows in column 2
ax1 = fig.add_subplot(gs[:, 2])  # Spans both rows in column 3
ax4_top = fig.add_subplot(gs[0, 3])  # Top plot in column 4
ax4_bottom = fig.add_subplot(gs[1, 3], sharex=ax4_top)  # Bottom plot in column 4, shares x-axis

# Adjust layout with increased padding to add space between subplots
plt.tight_layout()

ax1.set_xlabel(r'$\omega$', labelpad=-1)
ax2.set_xlabel(r'$N$', labelpad=-1)
ax3.set_xlabel(r'$N$', labelpad=-1)
ax4_top.set_xlabel(r'$N$', labelpad=-1)
ax4_bottom.set_xlabel(r'$N$', labelpad=-1)

ax1.set_ylabel(r'$S(\omega)$', labelpad=-3)
ax2.set_ylabel(r'$\sqrt{N T} \Delta b_{\text{est}}$', labelpad=-3)
ax3.set_ylabel(r'$r$', labelpad=-3)
ax4_top.set_ylabel(r'$\kappa_{\text{opt}}$', labelpad=-3)
ax4_bottom.set_ylabel(r'$\tau_{\text{opt}}$', labelpad=-3)

# set scaling for x-axis
ax1.set_xscale('log')
ax2.set_xscale('log')
ax3.set_xscale('log')
ax4_top.set_xscale('log')
ax4_bottom.set_xscale('log')

#set limits for x-axis
ax1.set_xlim((0.1, 10))
ax2.set_xlim((1, 10**5))
ax3.set_xlim((1, 10**5))
ax4_bottom.set_xlim((1, 10**5))
ax4_top.set_xlim((1, 10**5))

#set scaling for y-axis
ax1.set_yscale('log')
ax2.set_yscale('linear')
ax3.set_yscale('log')
ax4_top.set_yscale('log')
ax4_bottom.set_yscale('log')

#set limits for y-axis
ax1.set_ylim((0.1, 3))
ax1.set_yticks([0.1, 1])
ax1.set_yticklabels(['.1', '1'])
ax2.set_yticks([0.01, 1, np.sqrt(np.e)])
ax2.set_yticklabels(['0', '1', r'$\sqrt{e}$'])
ax2.set_ylim((0, 1.8))
ax2.set_yticks([0, 1, np.sqrt(np.e)])
ax2.set_yticklabels(['0', '1', r'$\sqrt{e}$'])
ax3.set_ylim((1, 10))
ax3.set_yticks([1, np.sqrt(np.e), 10.001])
ax3.set_yticklabels(['1', r'$\sqrt{e}$', '10'])
ax3.get_yticklabels()[1].set_color('gray')  # Set the color of the sqrt(e) label to gray
ax4_bottom.set_yticks([0.01, 0.1, 1])
ax4_bottom.set_yticklabels(['.01', '.1', '1'])
ax4_top.set_ylim((0.01, 2))
ax4_top.set_yticks([0.01, 0.1, 1])
ax4_top.set_yticklabels(['.01', '.1', '1'])

label_noise = ['Gaussian', 'White', 'Linear', 'Ohmic']

# Ploting
# Load data
step = 1
out = np.load(name_file)
for idt, t in enumerate(T):
    normalization = np.sqrt(1 / (t * N[::step]))
    w = np.logspace(-1, 1, 1001)
    tau = np.linspace(0, 10, 1001)
    ax1.plot(w, temp_spec[idt](w), c=c[idt], lw=1.5, ls='-', zorder=-1)
    ax2.plot(N[::step], np.sqrt(out[idt, ::step, 0]) / normalization, marker=m[0], c=c[idt], ls=ls[0], zorder = 1, alpha=alpha[0])
    ax2.plot(N[::step], np.sqrt(out[idt, ::step, 3]) / normalization, marker=m[1], c=c[idt], ls=ls[1], zorder= 0, alpha= alpha[1])     
    ax3.plot(N[::step], np.sqrt(out[idt, ::step, 3]) / np.sqrt(out[idt, ::step, 0]), label= label_noise[idt], marker=m[0], ls=ls[0], c=c[idt], zorder= 1, alpha=alpha[0])
    ax4_top.plot(N[::step], out[idt, ::step, 1], marker=m[0], c=c[idt], ls=ls[0], zorder= 5 - idt//2, alpha=alpha[0])
    ax4_bottom.plot(N[::step], out[idt, ::step, 2], marker=m[0], c=c[idt], ls=ls[0], zorder= 5 - idt//2, alpha=alpha[0])
    ax4_bottom.plot(N[::2*step], out[idt, ::2*step, 5], marker=m[1], c=c[idt], ls=ls[1], zorder= 1 - idt//2, alpha=alpha[1])
    if idt == 2:
        ax4_top.plot(N[::2 * step], out[idt, ::2 * step, 1], marker=m[0], c=c[idt], ls=ls[0], zorder= 5 + idt//2, alpha=alpha[0])
        ax4_bottom.plot(N[::2 * step], out[idt, ::2 * step, 2], marker=m[0], c=c[idt], ls=ls[0], zorder= 5 + idt//2, alpha=alpha[0])
        ax4_bottom.plot(N[::4 * step], out[idt, ::4 * step, 5], marker=m[1], c=c[idt], ls=ls[1], zorder= 1 + t//2, alpha=alpha[1])
    if idt == 3:
        ax4_top.plot(N[::2 * step], out[idt, ::2 * step, 1], marker=m[0], c=c[idt], ls=ls[0], zorder= 5 + idt//2, alpha=alpha[0])
        ax4_bottom.plot(N[::2 * step], out[idt, ::2 * step, 2], marker=m[0], c=c[idt], ls=ls[0], zorder= 5 + idt//2, alpha=alpha[0])
        ax4_bottom.plot(N[::4 * step], out[idt, ::4 * step, 5], marker=m[1], c=c[idt], ls=ls[1], zorder= 1 + idt//2, alpha=alpha[1])

ax3.axhline(y=np.sqrt(np.e), color='gray', linestyle='-', lw=0.5, zorder=0, alpha=0.5)
ax4_top.plot(N, np.ones(len(N)), color='gray', linestyle=ls[1], zorder=0, alpha=0.7)
ax2.axhline(y=1, color='gray', linestyle='-', zorder=0, lw= 0.5, alpha=0.7)

# Add a text label parallel to the line
rotation_angle = 46
ax3.text(10**3, 0.8 * (10**(3))**(1/6)+0.8, r'$\propto N^{1/6}$', color='gray', rotation=rotation_angle, fontsize=8, va='center', ha='left')
ax3.plot(N, 0.8 * (N)**(1/6), color='gray', linestyle='-', zorder=0, alpha=0.7, lw=1)

rotation_angle = 54
ax3.text(10**3, 0.8 * (10**(3))**(1/4)+ 2, r'$\propto N^{1/4}$', color='gray', rotation=rotation_angle, fontsize=8, va='center', ha='left')
ax3.plot(N, 0.8 * (N)**(1/4), color='gray', linestyle='-', zorder=0, alpha=0.7, lw=1)

rotation_angle = -40
ax4_top.text(10**2, 0.9 * (10**(2))**(-1/3), r'$\propto N^{-1/3}$', color='gray', rotation=rotation_angle, fontsize=8, va='center', ha='left')
ax4_top.plot(N, 1.6 * (N)**(-1/3), color='gray', linestyle='-', zorder=0, alpha=0.7, lw=1)

rotation_angle = -42
ax4_top.text(10**1, 0.12 * (10**(1))**(-3/8), r'$\propto N^{-3/8}$', color='gray', rotation=rotation_angle, fontsize=8, va='center', ha='left')
ax4_top.plot(N, 0.8 * (N)**(-3/8), color='gray', linestyle='-', zorder=0, alpha=0.7, lw=1)

rotation_angle = -36
ax4_bottom.text(10**2.5, 1.14 * (10**(2.5))**(-1/3), r'$\propto N^{-1/3}$', color='gray', rotation=rotation_angle, zorder= 10, fontsize=8, va='center', ha='left')
ax4_bottom.plot(N, 2.05 * (N)**(-1/3), color='gray', linestyle='-', zorder=0, alpha=0.7, lw=1)

rotation_angle = -32
ax4_bottom.text(10**1, 0.1 * (10**(1))**(-1/4), r'$\propto N^{-1/4}$', color='gray', rotation=rotation_angle, fontsize=8, va='center', ha='left')
ax4_bottom.plot(N, 0.45 * (N)**(-1/4), color='gray', linestyle='-', zorder=0, alpha=0.7, lw=1)

ax2.plot(10, 10, marker=m[0], c='gray', ls=ls[0], label='Squeezed')
ax2.plot(10, 10, marker=m[1], c='gray', ls=ls[1], label='Separable')       
leg2 = ax2.legend(frameon=True, handlelength=1.5, handletextpad=0.5, borderpad=0.5, borderaxespad=0.5)
leg3 = ax3.legend(frameon=True, handlelength=1.5, handletextpad=0.5, borderpad=0.5, borderaxespad=0.5)
leg2.get_frame().set_linewidth(0.5)
leg3.get_frame().set_linewidth(0.5)


# Add bold letters (a)-(e) to the bottom right of each subplot
labels = ['c','a', 'b', 'd', 'e']
axes = [ax1, ax2, ax3, ax4_top, ax4_bottom]
for ax, label in zip(axes[:3], labels[:3]):
    if label == 'c':
        ax.text(0.98, 0.005, label, transform=ax.transAxes, ha='right', va='bottom', fontweight='bold', zorder=10)
    else:
        ax.text(0.99, 0.005, label, transform=ax.transAxes, ha='right', va='bottom', fontweight='bold', zorder=10)


for ax, label in zip(axes[3:], labels[3:]):
    ax.text(0.02, 0.01, label, transform=ax.transAxes, ha='left', va='bottom', fontweight='bold', zorder=10)

plt.savefig('fig3.svg', bbox_inches='tight')
plt.show()

