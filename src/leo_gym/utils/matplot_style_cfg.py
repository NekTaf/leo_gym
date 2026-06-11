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
plt.style.use(['science','default'])

plt.rcParams['figure.dpi'] = 300
plt.rcParams['axes.xmargin'] = 0

fontsize = 10 

mpl.rcParams.update({
    "text.usetex": False,         
    "font.family": "serif",

    "font.size": 10 ,
    "axes.labelsize": 10 ,
    "axes.titlesize": 10 ,
    "legend.fontsize": 9 ,
    "xtick.labelsize": 9 ,
    "ytick.labelsize": 9 ,

    "lines.linewidth": 1.5,
    "axes.linewidth": 0.8,
    "xtick.major.width": 0.8,
    "ytick.major.width": 0.8,
    "xtick.major.size": 3.5,
    "ytick.major.size": 3.5,

    # "lines.linewidth": 1.5,
    # "axes.linewidth": 0.8,
    "legend.frameon": False,
    "axes.grid": True,
})
