"""
Import this file for defining common matplot plotting style

"""

# Third-party
import matplotlib as mpl
import matplotlib.pyplot as plt
import scienceplots
from IPython.display import display, HTML


# display(HTML(
#     '<script type="text/javascript" async '
#     'src="https://cdnjs.cloudflare.com/ajax/libs/mathjax/2.7.1/MathJax.js'
#     '?config=TeX-MML-AM_SVG"></script>'
# ))

# plt.style.use(['science','ieee','high-vis'])
plt.style.use(['science','bmh'])

plt.rcParams['figure.dpi'] = 300

fig_width_in_inches = 5
textwidth_pt = 384
fontsize = 10 * (fig_width_in_inches / (textwidth_pt / 72.27))

mpl.rcParams.update({
    
    "text.usetex": False,           
    # "pgf.texsystem": "pdflatex",   
    # "font.family": "serif",         
    # "font.serif": ["Times New Roman"], # Set your desired font
    "font.family": "serif",
    "pgf.rcfonts": False,
    # "mathtext.fontset": "stix",

    "axes.labelsize": fontsize,          
    "font.size": fontsize,
    "legend.fontsize": fontsize,
    "xtick.labelsize": fontsize,
    "ytick.labelsize": fontsize,
    "axes.xmargin": 0
})