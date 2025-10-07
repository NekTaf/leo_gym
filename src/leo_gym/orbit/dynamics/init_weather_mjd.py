# Standard library
import os

# Third-party
import numpy as np
import spiceypy as spice

from .dynamics import Dynamics


def load_spice_kernels()->None:
    original_cwd = os.getcwd()
    try:
        os.chdir(os.path.dirname(os.path.abspath(__file__)))
        os.chdir("..")
                
        # for JPL Spice
        total_kernels_loaded = spice.ktotal('ALL')

        if total_kernels_loaded == 0:
            spice.furnsh("spice_kernels/Kernels/kernels_to_load.txt")

        else:
            # print("Spice kernels already loaded....")
            pass
            
    finally:
        os.chdir(original_cwd)

