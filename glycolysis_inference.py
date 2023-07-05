# handy-dandy
import os
import sys
from tqdm import tqdm

dataPath='MODEL1303260011_pt10.csv'

FOLDER_NAME = 'glycolysis/'
NOISE=False
DATA_OMISSION_CODE = 'E'
ADVI_ITERATIONS = 100000 # 60000
FAIL_LOG_FILE = f'failed-{DATA_OMISSION_CODE}.log'
DATA_FOLDER_NAME = 'generated_data/'
name_of_script = sys.argv[1]
LEARNING_RATE = 1E-2



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

# biochemical pathway simulators
import cobra
import tellurium as te

# plotting
import matplotlib.pyplot as plt
import seaborn as sns
sns.set(context='talk', style='ticks',
        color_codes=True, rc={'legend.frameon': False})
import arviz as az

import cloudpickle

# linlog Bayesian Metabolic Control Analysis
import emll
from emll.util import initialize_elasticity

itr = ADVI_ITERATIONS
folder_name = FOLDER_NAME

r = te.loada("MODEL1303260011.ant")
s = te.loada("MODEL1303260011_cobra.ant")
with open("temp.txt", "w") as f:
    f.write(s.getSBML())

model = cobra.io.read_sbml_model("temp.txt")
os.remove("temp.txt") 

data = pd.read_csv(dataPath).astype(float)

N = r.getFullStoichiometryMatrix()
nm, nr = N.shape

n_exp = len(data)   

e_cols = [col for col in data.columns if 'ENZYME' in col]
x_cols = r.getFloatingSpeciesIds()    
exclude = ['AMP', 'NADH', 'UDG']
y_cols = [y for y in r.getBoundarySpeciesIds() if y not in exclude]
v_cols = r.getReactionIds()

e = data[e_cols]
x = data[x_cols]
y = data[y_cols]
v = data[v_cols]

# the reference index is the strain that produces the most of a 
# desired product here, we arbitrarily choose a random species as 
# our desired product
# target_rxn_i = -1
desired_product = 'succinate_branch' # y_cols[target_rxn_i] # the boundary form of the desired product
# the strain index at which the level of desired product is highest
ref_ind = data.idxmax()['EX_' + desired_product] 
# list of boundary reactions
exRxns = [i.id for i in model.reactions if 'EX' in i.id] 
# boundary reaction that produces desired product
target_rxn = [i for i in exRxns if desired_product == i[3:]][0]
desired_product = target_rxn[3:] # the internal form of the desired product

e_star = e.iloc[ref_ind].values
x_star = x.iloc[ref_ind].values
y_star = y.iloc[ref_ind].values
v_star = v.iloc[ref_ind].values

e_star[e_star == 0] = 1e-6
v_star[v_star == 0] = 1e-9
y_star[y_star == 0] = 1e-6
y[y <= 0] = 1e-6

assert (len(e.values[e.values <= 0]) == 0)
assert (len(y.values[y.values <= 0]) == 0)
assert (len(e_star[e_star <= 0]) == 0)
assert (len(y_star[y_star <= 0]) == 0)

# Normalize to reference values (and drop trivial measurement)
en = e.divide(e_star)
xn = x.divide(x_star)
yn = y.divide(y_star)
vn = v.divide(v_star)

# get rid of any 0 values
vn[vn <= 0] = 1e-6 # need to drop Nan values
en[en <= 0] = 1e-6
yn[yn <= 0] = 1e-6

en = en.drop(ref_ind)
xn = xn.drop(ref_ind)
yn = yn.drop(ref_ind)
vn = vn.drop(ref_ind)      

# Correct negative flux values at the reference state
N[:, v_star < 0] = -1 * N[:, v_star < 0]
v_star = np.abs(v_star)

# Correct all 0 fluxes. Cannot divide nor take log of 0
v_star[v_star == 0] = 1e-9

Ex = emll.create_elasticity_matrix(model)
Ey = np.zeros((nr, len(y_cols))) # (reactions, number of external species)

# external species reaction indices
# aka what is the index of external species reactions?
exSp = [model.reactions.index(i.id) for i in model.reactions if 'EX' in i.id]
for i in range(len(exSp)): 
    Ey[int(exSp[i]), i] = 1

ll = emll.LinLogLeastNorm(N, Ex, Ey, v_star)

### creating the probability model
with pm.Model() as pymc_model:
    
    # Initialize elasticities
    Ex_t = pm.Deterministic('Ex', initialize_elasticity(N, 'ex', b=0.05, sd=1, alpha=5))
    Ey_t = pm.Deterministic('Ey', initialize_elasticity(-Ey.T, 'ey', b=0.05, sd=1, alpha=5))

with pymc_model:
    yn_t = pm.Normal('xn_t', mu=1, sigma=10, shape=yn.shape, 
                     initval=0.1 * np.random.randn(yn.shape[0], yn.shape[1]))
    
    # Error priors
    y_err = pm.HalfNormal('y_error', sigma=0.05, initval=.01)

    # Calculate steady-state concentrations and fluxes from elasticities
    chi_ss, vn_ss_x = ll.steady_state_aesara(Ex_t, Ey_t, en.to_numpy(), yn_t)
    # y_ss, vn_ss_y = ll.steady_state_aesara(Ey_t, Ex_t, en.to_numpy(), xn.to_numpy())

    # Error distributions for observed steady-state concentrations and fluxes
    
    v_hat_obs = pm.Normal('v_hat_obs', mu=vn_ss_x, sigma=0.1, observed=vn) # both bn and v_hat_ss are (28,6)
    chi_obs = pm.Normal('chi_obs', mu=chi_ss, sigma=0.1, observed=xn) # chi_ss and xn is (28,4)
    # y_obs = pm.Normal('y_obs', mu=y_ss, sigma=y_err, observed=yn)
    e_obs = pm.Normal('e_obs', mu=1, sigma=0.1, observed=en)

with pymc_model:
    trace_prior = pm.sample_prior_predictive() 

# sampling
with pymc_model:
    approx = pm.ADVI()
    hist = approx.fit(n=itr, obj_optimizer=pm.adagrad_window(learning_rate=LEARNING_RATE), obj_n_mc=1)

with pymc_model:
    trace = hist.sample(1000)    
    ppc_vi = pm.sample_posterior_predictive(trace, random_seed=1)

# label
m_labels = [m.id for m in model.metabolites]
r_labels = [r.id for r in model.reactions]
y_labels = y_cols

ex_labels = np.array([['$\epsilon_{' + '{0},{1}'.format(rlabel, mlabel) + '}$'
                    for mlabel in m_labels] for rlabel in r_labels]).flatten()
ey_labels = np.array([['$\epsilon_{' + '{0},{1}'.format(rlabel, mlabel) + '}$'
                    for mlabel in y_labels] for rlabel in r_labels]).flatten()

e_labels = np.hstack((ex_labels, ey_labels))


cloudpickle.dump({'advi': approx,
'approx': approx,
'trace': trace,
'trace_prior': trace_prior,
'ppc': ppc_vi,
'e_labels': e_labels,
'r_labels': r_labels,
'm_labels': m_labels,
'y_labels': y_labels,
'll': ll}, file=open(f'folder_name + {name_of_script}_advi.pgz', "wb"))