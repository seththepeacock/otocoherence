"Init"

from helper_funcs import *
import helper_funcs
import importlib

importlib.reload(helper_funcs)
import matplotlib.pyplot as plt

plt.rcParams["svg.fonttype"] = "none"
from collections import defaultdict
import numpy as np
import pandas as pd

dirs = get_dirs(root="C:\\Users\\setht\\Dropbox\\Citadel\\GitHub\\otocoherence")
dpi = 1000