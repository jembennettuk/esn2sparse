import numpy as np
import sympy
import torch
import matplotlib
import os
def seedNumpyRng(seed):
    # Seed Numpy RNG
    s = list(sympy.primerange(11111,33333))
    np.random.default_rng(s[seed])

def seedTorchRng(seed):
    # Seed Torch RNG
    s = list(sympy.primerange(11111,33333))
    torch.manual_seed(s[seed])

def setupMatplotlib():
    # Set Matplotlib style
    matplotlib.rcParams['savefig.dpi'] = 300
    matplotlib.rcParams.update({'font.size': 6})
    matplotlib.rcParams['svg.fonttype'] = 'none'
    matplotlib.rcParams['savefig.format'] = 'svg'
    matplotlib.rcParams['font.family'] = 'sans-serif'

def checkMakeDir(dirName):
    if not os.path.exists(dirName):
        os.mkdir(dirName)