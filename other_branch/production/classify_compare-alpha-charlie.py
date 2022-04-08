'''
    The file compares the results from SWARM Alpha and Charlie

    It assesses the EPBs they have captured and determines if they 
    are seeing the same thing. 

    If Alpha and Charlie disagree, the EPB is dropped

    Created by Sachin A. Reddy

    March 2022.
'''

import pandas as pd
from pathlib import Path
from datetime import date
import numpy as np
from matplotlib import pyplot as plt
from matplotlib import gridspec
import seaborn as sns
pd.set_option('display.max_rows', None) #or 10 or None