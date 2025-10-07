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
plt.style.use(['science','ggplot'])

plt.rcParams['figure.dpi'] = 600

mpl.rcParams.update({
    "text.usetex": False,           # Use LaTeX for all text
    # "pgf.texsystem": "pdflatex",   # Use pdflatex for LaTeX rendering
    "font.family": "serif",        # LaTeX-compatible serif font
    "font.serif": ["DejaVu Sans"], # Set your desired font
    "axes.labelsize": 12,          # Set label size
    "font.size": 12,
    "legend.fontsize": 10,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
})