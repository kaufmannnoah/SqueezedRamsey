from matplotlib.gridspec import GridSpec
import matplotlib.pyplot as plt

def set_plot_styles():
    plt.rcParams.update({
        'font.size': 8,          # Set font size
        'axes.titlesize': 9,     # Title font size
        'axes.labelsize': 8,     # Axis label font size
        'xtick.labelsize': 7,     # X-tick label font size
        'ytick.labelsize': 7,     # Y-tick label font size
        'lines.linewidth': 1,   # Line width
        'lines.markersize': 1.5,    # Marker size
        'axes.linewidth': 1,      # Axis line width
        'legend.fontsize': 8,     # Legend font size
        'figure.dpi': 300,        # Figure resolution
    })

set_plot_styles()
figures = [
    {'text': 'a', 'kwargs': {'fontsize': 9, 'fontweight': 'bold'}},
    {'text': 'b', 'kwargs': {'fontsize': 9, 'fontweight': 'bold'}},
    {'text': r'$R_{-\pi/2}$', 'kwargs': {}},
    {'text': r'$R_{\pi/2}$', 'kwargs': {}},
    {'text': r'$1$', 'kwargs': {}},
    {'text': r'$2$', 'kwargs': {}},
    {'text': r'$N$', 'kwargs': {'fontweight': 'bold'}},
    {'text': r'$\tau$', 'kwargs': {}},
    {'text': r'$\tau$', 'kwargs': {}},
    {'text': r'$\tau$', 'kwargs': {}},
    {'text': 'A', 'kwargs': {}},
    {'text': 'B', 'kwargs': {}},
    {'text': 'C', 'kwargs': {}},
    {'text': 'D', 'kwargs': {}},
    {'text': 'E', 'kwargs': {}},
    {'text': r'$T/\tau$', 'kwargs': {}},
    {'text': r'$J_x$', 'kwargs': {}},
    {'text': r'$J_y$', 'kwargs': {}},
    {'text': r'$J_z$', 'kwargs': {}},
    {'text': r'$b$', 'kwargs': {'color': '#7F7F7F'}},
    {'text': 'squeezing', 'kwargs': {}},
    {'text': 'free\nevolution', 'kwargs': {}}  # free evolution on two lines
]

for i, item in enumerate(figures):
    if i < len(figures)-3:
        fig, ax = plt.subplots(figsize=(0.3, 0.3))
    else:
        fig, ax = plt.subplots(figsize=(1, 0.3))
    ax.axis('off')
    ax.set_facecolor('none')
    fig.patch.set_facecolor('none')
    ax.text(0.15, 0.15, item['text'], va='center', ha='center', transform=ax.transAxes, **item['kwargs'])
    plt.tight_layout()
    plt.savefig(f"label_{i+1}.svg", format="svg", bbox_inches='tight')
    plt.close(fig)