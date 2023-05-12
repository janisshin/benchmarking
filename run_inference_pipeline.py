FOLDER_NAME = '10sp_w-Allo/'
NOISE=False
DATA_OMISSION_CODE = 'A-x'
ADVI_ITERATIONS = 60000 # 30000
FAIL_LOG_FILE = f'failed10-{DATA_OMISSION_CODE}.log'
# what position is the passNumber? e.g. 1 for "data_24_pt10.csv or 2
# for "mass_action_152.xml" 
passN = 1 

######################################################################

# handy-dandy
import os
import sys
import re
from tqdm import tqdm
import warnings
# warnings.filterwarnings("error")
# warnings.resetwarnings()
# warnings.filterwarnings('ignore')
from datetime import datetime;

# arrays/dataframes
import numpy as np
np.random.seed(0)
np.set_printoptions(threshold=sys.maxsize)

import pandas as pd
pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)

from csv import writer

# math/stats
import scipy
import scipy.stats
import pymc as pm
import aesara
import arviz as az

# biochemical pathway simulators
import cobra
import tellurium as te

# plotting
import matplotlib.pyplot as plt
import seaborn as sns
sns.set(context='talk', style='ticks',
        color_codes=True, rc={'legend.frameon': False})

# linlog Bayesian Metabolic Control Analysis
import emll
from emll.util import initialize_elasticity

import inference_utils as bi

######################################################################

# make results folders 
results_dir = './' + FOLDER_NAME + 'results/' + DATA_OMISSION_CODE + '/'

folders_to_create = [results_dir,
                     results_dir + 'convergence/',
                     results_dir + 'elast-hdi/',
                     results_dir + 'elast-plot/',
                     results_dir + 'FCC-hdi/',
                     results_dir + 'FCC-graph/',
                     results_dir + 'MCC-hdi/',
                     results_dir + 'MCC-graph/']

for dir in folders_to_create:
    try: 
        os.mkdir(dir)
    except:
        pass

if NOISE:
    DATA_FOLDER_NAME = 'noisy_generated_data/'
else:
    DATA_FOLDER_NAME = 'generated_data/'

for dataPath in os.listdir(FOLDER_NAME + DATA_FOLDER_NAME):
    modelNo = re.split(r'[_|.]', dataPath)[passN]
    path = f'mass_action_{modelNo}.ant'
    print(path)
    with open(FAIL_LOG_FILE, "a") as f:
            f.write(str(datetime.now())+'\n')

    try:
        bi.run_analysis(path, dataPath, itr=ADVI_ITERATIONS, folder_name=FOLDER_NAME, noise=NOISE)        
    except: 
        with open(FAIL_LOG_FILE, "a") as f:
            f.write(dataPath + '\n')
    