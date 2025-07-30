from matplotlib.gridspec import GridSpec
import matplotlib.pyplot as plt
from AAfunctions.fun_clean import *
from param import *


# Define a function to set consistent plot styles
set_plot_styles()

c = colors()

ls_2=['-', '-', '-', '-']

# Apply the styles
set_plot_styles()

# Define figure dimensions (7 inches wide for PRX Quantum two-column layout)
fig_width = 3.375  # in inches
fig_height = fig_width # Aspect ratio for a visually pleasing layout
fig = plt.figure(figsize=(fig_width, fig_height))

# Define the grid layout
width_ratios = [1, 1]
height_ratios = [2, 2]
gs = GridSpec(2, 2, width_ratios=width_ratios, height_ratios=height_ratios, figure=fig)
ax1_top = fig.add_subplot(gs[0, 0])  # Spans both rows in column 1
ax1_bottom = fig.add_subplot(gs[0, 1])  # Spans both rows in column 1
ax2 = fig.add_subplot(gs[1, :])  # Spans both rows in column 2
plt.tight_layout()

#set axis labels
ax1_top.set_xlabel(r'$\omega/\sigma$', labelpad=-1)
ax1_top.set_ylabel(r'$S(\omega)$', labelpad = 4)

ax1_bottom.set_xlabel(r'$T\sigma$', labelpad=-1)
ax1_bottom.set_ylabel(r'$\gamma(T)/T$', labelpad=-1)

ax2.set_xlabel(r'$T\sigma$', labelpad=-1)
ax2.set_ylabel(r'$\Delta b_{\text{est}}$', labelpad= 0)

label_noise = ['Gaussian', 'White', 'Linear', 'Ohmic']

#set limits
ax1_top.set_yscale('log')
ax1_top.set_ylim((0.4, 18))
ax1_top.set_yticks([1, 10])
#ax1_top.set_yticklabels(['1', '10'])
ax1_top.set_xscale('log')
ax1_top.set_xlim((0.1, 10))
ax1_top.set_xticks([0.1, 1, 10])
#ax1_top.set_xticklabels(['.1', '1', '10'])

ax1_bottom.set_yscale('log')
ax1_bottom.set_ylim((0.4, 18))
ax1_bottom.set_yticks([1, 10])
#ax1_bottom.set_yticklabels(['1', '10'])
ax1_bottom.set_xscale('log')
ax1_bottom.set_xlim((.1, 100))
ax1_bottom.set_xticks([.1, 1, 10, 100])
#ax1_bottom.set_xticklabels(['.1', '1', '10', '100'])

ax2.set_yscale('log')
ax2.set_ylim((0.0004, 0.7))
ax2.set_yticks([0.001, 0.01, 0.1])
#ax2.set_yticklabels(['.001', '.01', '.1'])
ax2.set_xscale('log')
ax2.set_xlim((.1, 1000))
ax2.set_xticks([.1, 1, 10, 100, 1000])
#ax2.set_xticklabels(['.1', '1', '10', '10', '1000'])

# Ploting
# Load data
step = 1
out = np.load(name_file)
for t in range(4):
    TT = TT_list[t]
    w = np.logspace(-1, 1, 1001)
    tau = np.linspace(.1, 100, 1001)
    ax1_top.plot(w, temp_spec[t](w*si_ar[t]), c=c[t], lw= 1.5, ls=ls_2[t], zorder= -1)
    ax1_top.plot(w, temp_spec[t](w*si_ar[t]), c=c[t], lw= 1.5, ls=ls_2[t], alpha= 0.7, zorder= (t+1)%2)
    ax1_bottom.plot(tau, [temp_cor[t](tau_i/si_ar[t], 0)/(tau_i/si_ar[t]) for tau_i in tau], c=c[t], ls=ls_2[t], lw= 1.5, zorder= -1)
    ax1_bottom.plot(tau, [temp_cor[t](tau_i/si_ar[t], 0)/(tau_i/si_ar[t]) for tau_i in tau], c=c[t], lw= 1.5, ls=ls_2[t], alpha= 0.7, zorder= (t+1)%2)
    ax2.plot(TT[::step]*si_ar[t], np.sqrt(out[t, ::step, 0] / A_ar[t]), c=c[t], lw= 1.5, zorder= -1, label=label_noise[t], ls=ls_2[t])
    ax2.plot(TT[::step]*si_ar[t], np.sqrt(out[t, ::step, 0] / A_ar[t]), c=c[t], lw= 1.5, alpha= 0.7, zorder= (t+1)%2, ls=ls_2[t])    
leg = ax2.legend(handlelength=1.7)
leg.get_frame().set_linewidth(0.5)

labels = ['a', 'b', 'c']
axes = [ax1_top, ax1_bottom, ax2]
for ax, label in zip(axes, labels):
    if label == 'c':
        ax.text(0.99, 1 - 0.02, label, transform=ax.transAxes, ha='right', va='top', fontweight='bold', zorder=10)
    else:
        ax.text(0.98, 0.98, label, transform=ax.transAxes, ha='right', va='top', fontweight='bold', zorder=10)

# Show the plot
plt.savefig('fig5.pdf', bbox_inches='tight')
plt.show()

