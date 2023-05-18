"""This notebook will run BMCA analysis for a batch of models. 
This notebook is designed to run within the framework of the following files:
- `generating_synthetic_models.ipynb`
- `generated_model_analysis.ipynb`
"""
# handy-dandy
import os
import sys
from tqdm import tqdm

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

######################################################################

def run_analysis(path, dataPath, itr=30000, folder_name="./", results_dir='./results/', noise=False):
    """
    path = name of Antimony model file (the cobra compatible version of model)
    dataPath = filepath of dataset (including dataset name)
    itr = number of iterations in ADVI
    folder_name = where to pull model data and store results
    """
    r = te.loada(folder_name + 'antimony/' + path)
    with open("temp.txt", "w") as f:
        f.write(r.getSBML())

    model = cobra.io.read_sbml_model("temp.txt")
    os.remove("temp.txt") 
    
    # return r, model
    if noise: 
        data = pd.read_csv(folder_name + 'noisy_generated_data/' + 
                           dataPath).astype(float)
    else: 
        data = pd.read_csv(folder_name + 'generated_data/' + 
                           dataPath).astype(float)

    N = r.getFullStoichiometryMatrix()
    nm, nr = N.shape

    n_exp = len(data)   

    e_cols = [col for col in data.columns if 'E' in col]
    x_cols = r.getFloatingSpeciesIds()    
    y_cols = [col for col in data.columns if 'B' in col]
    v_cols = [col for col in data.columns if 'flux' in col]

    e = data[e_cols]
    x = data[x_cols]
    y = data[y_cols]
    v = data[v_cols]

    # the reference index is the strain that produces the most of a 
    # desired product here, we arbitrarily choose a random species as 
    # our desired product
    target_rxn_i = -1
    desired_product = y_cols[target_rxn_i] # the boundary form of the desired product
    # the strain index at which the level of desired product is highest
    ref_ind = data.idxmax()[desired_product] 
    # list of boundary reactions
    exRxns = [i.id for i in model.reactions if 'EX' in i.id] 
    # boundary reaction that produces desired product
    target_rxn = [i for i in exRxns if desired_product[1:] == i[4:]][0]
    desired_product = target_rxn[3:] # the internal form of the desired product
        
    e_star = e.iloc[ref_ind].values
    x_star = x.iloc[ref_ind].values
    y_star = y.iloc[ref_ind].values
    v_star = v.iloc[ref_ind].values

    v_star[v_star == 0] = 1e-9
    y_star[y_star == 0] = 1e-6

    # Normalize to reference values (and drop trivial measurement)
    en = e.divide(e_star)
    xn = x.divide(x_star)
    yn = y.divide(y_star)
    vn = v.divide(v_star)

    en = en.drop(ref_ind)
    xn = xn.drop(ref_ind)
    yn = yn.drop(ref_ind)
    vn = vn.drop(ref_ind)    

    if noise:
        model.objective = target_rxn
        for i, rxn in enumerate(model.reactions):
            if 'EX' in rxn.id:
                model.reactions.get_by_id(rxn.id).upper_bound = v_star[i]
                model.reactions.get_by_id(rxn.id).lower_bound = v_star[i]
            elif v_star[i] > 0:
                model.reactions.get_by_id(rxn.id).upper_bound = v_star[i] * 2
                model.reactions.get_by_id(rxn.id).lower_bound = 0        
            else:
                model.reactions.get_by_id(rxn.id).upper_bound = 0
                model.reactions.get_by_id(rxn.id).lower_bound = v_star[i] * 2        
        sol = model.optimize()
        v_star = sol.fluxes.values
        
        # if solver status is infeasible
        if not np.allclose(N @ v_star, 0):
            for r in model.reactions:
                if 'EX' not in r.id:
                    r.upper_bound = 1000
                    r.lower_bound = -1000
            sol = model.optimize()
            new_v_star = sol.fluxes.values
        # solver status may still be infeasible, but then the model will just 
        # fail out of the pipeline    

    N[:, v_star < 0] = -1 * N[:, v_star < 0]
    v_star = np.abs(v_star)

    Ex = emll.create_elasticity_matrix(model)
    Ey = np.zeros((nr, len(y_cols))) # (reactions, number of external species)
    
    # external species reaction indices
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
        # Error priors. 
        #v_err = pm.HalfNormal('v_error', sigma=0.05, initval=.1)
        #x_err = pm.HalfNormal('x_error', sigma=0.05, initval=.1)
        #y_err = pm.HalfNormal('y_error', sigma=0.05, initval=.01)
        #e_err = pm.HalfNormal('e_error', sigma=10, initval=.01)
        
        # Calculate steady-state concentrations and fluxes from elasticities
        chi_ss, vn_ss_x = ll.steady_state_aesara(Ex_t, Ey_t, en.to_numpy(), yn)
        y_ss, vn_ss_y = ll.steady_state_aesara(Ey_t, Ex_t, en.to_numpy(), xn)

        # Error distributions for observed steady-state concentrations and fluxes
        
        v_hat_obs = pm.Normal('v_hat_obs', mu=vn_ss_x, sigma=0.1, observed=vn) # both bn and v_hat_ss are (28,6)
        chi_obs = pm.Normal('chi_obs', mu=chi_ss, sigma=0.1, observed=xn) # chi_ss and xn is (28,4)
        y_obs = pm.Normal('y_obs', mu=y_ss, sigma=0.1, observed=yn)
        e_obs = pm.Normal('e_obs', mu=1, sigma=0.1, observed=en)

    with pymc_model:
        trace_prior = pm.sample_prior_predictive() 
    
    # sampling
    with pymc_model:
        approx = pm.ADVI()
        hist = approx.fit(n=itr, obj_optimizer=pm.adagrad_window(learning_rate=5E-3), obj_n_mc=1)

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
    
    # generating and storing results
    dataset_name = dataPath.split(".")[0]

    plot_ADVI_converg(approx, itr, results_dir + 'convergence/', dataset_name)
    
    elasticities_to_csv(trace, e_labels, results_dir + 'elast-hdi/' + dataset_name)
    plot_elasticities(trace, trace_prior, N, e_labels, results_dir, dataset_name)
    mcc_df = ADVI_CCs_hdi(trace, trace_prior, 'mcc', r, ll, results_dir + 
                          f'MCC-hdi/{dataset_name}-MCC_hdi.csv', 
                          cc_indexer=desired_product)
    plot_CC_distbs(mcc_df, 'mcc', r, results_dir, dataset_name, cc_indexer=desired_product)
    
    fcc_df = ADVI_CCs_hdi(trace, trace_prior, 'fcc', r, ll, results_dir + 
                          f'FCC-hdi/{dataset_name}-FCC_hdi.csv', cc_indexer=target_rxn_i)
    plot_CC_distbs(fcc_df, 'fcc', r, results_dir, dataset_name, cc_indexer=target_rxn_i)
    """
    cloudpickle.dump({'advi': approx,
    'approx': approx,
    'trace': trace,
    'trace_prior': trace_prior,
    'e_labels': e_labels,
    'r_labels': r_labels,
    'm_labels': m_labels,
    'll': ll}, file=open(f'folder_name + {dataset_name}_advi.pgz', "wb"))
   """ 

def calculate_gt_FCCs(r, target_rxn_i):
    """
    objective: directly calculates the FCCs of model
    Parameters
    r: roadrunner object of model
    Returns FCCs for sink reaction
    """
    r.simulate(0,1000000)
    r.steadyState()
    return r.getScaledFluxControlCoefficientMatrix()[target_rxn_i]


def calculate_gt_MCCs(r, desired_product):
    """
    objective: directly calculates the CCCs of model
    Parameters
    r: roadrunner object of model
    Returns CCCs for sink reaction
    """
    r.simulate(0,1000000)
    r.steadyState()
    return r.getScaledConcentrationControlCoefficientMatrix()[desired_product]


def plot_ADVI_converg(approx, itr, results_dir, dataset_name):
    """
    objective: to plot the convergence of samples from ADVI 
    Parameters
    approx: the ADVI fitting
    itr: how many times to run ADVI
    results_dir: filepath of where to deposit r
    Returns nothing. Deposits an svg file of convergence plot 
    """
    with sns.plotting_context('notebook', font_scale=1.2):

        fig = plt.figure(figsize=(5,4))
        plt.plot(approx.hist + 30, '.', rasterized=True, ms=1)
        plt.yscale("log")
        # plt.ylim([1E5, 1E11])
        plt.xlim([0, itr])
        sns.despine(trim=True, offset=10)

        plt.ylabel('-ELBO')
        plt.xlabel('Iteration')
        plt.title(f'{dataset_name} ADVI convergence')
        plt.tight_layout()
        plt.savefig(results_dir + dataset_name + '-convergence.svg', transparent=True, dpi=200)


def elasticities_to_csv(trace, e_labels, results_dir):
    Ex_hdi = az.hdi(trace['posterior']['Ex'])['Ex'].to_numpy() #(13, 8, 2)
    Ey_hdi = az.hdi(trace['posterior']['Ey'])['Ey'].to_numpy() #(13, 2, 2)
    ex = Ex_hdi.reshape((Ex_hdi.shape[0]*Ex_hdi.shape[1],-1))
    ey = Ey_hdi.reshape((Ey_hdi.shape[0]*Ey_hdi.shape[1],-1))
    e_all = np.transpose(np.vstack([ex, ey]))
    e_df_vi = pd.DataFrame(e_all, columns=e_labels)
    e_df_vi.to_csv(results_dir + '-elasticities.csv')


def plot_elasticities(trace, trace_prior, N, e_labels, results_dir, dataset_name):
    
    # first relabel the indices
    e_guess = -N.T
    e_flat = e_guess.flatten()
    nonzero_inds = np.where(e_flat != 0)[0]
    zero_inds = np.where(e_flat == 0)[0]
    e_sign = np.sign(e_flat[nonzero_inds])
    flat_indexer = np.hstack([nonzero_inds, zero_inds]).argsort()

    exKinent = trace['posterior']['ex_kinetic_entries']
    exCapent = trace['posterior']['ex_capacity_entries']
    
    identifiable_elasticies = np.diff(az.hdi(exKinent)['ex_kinetic_entries'].to_numpy() < .75).flatten()
    
    elast_nonzero = pd.DataFrame((np.squeeze(exKinent.to_numpy()) * e_sign)[:, identifiable_elasticies],
                             columns=e_labels[nonzero_inds][identifiable_elasticies])
    null = pd.DataFrame(az.hdi(exCapent)['ex_capacity_entries'].to_numpy())
    sig = np.sign(null)[0] == np.sign(null)[1]

    elast_zero = pd.DataFrame(np.squeeze(exCapent.to_numpy())[:, sig], columns=e_labels[zero_inds[sig]])
    elast_posterior = elast_nonzero.iloc[:, elast_nonzero.mean().argsort()].join(elast_zero)

    prior_Es = np.dstack((np.squeeze(trace_prior['prior']['Ex'].to_numpy()), np.squeeze(trace_prior['prior']['Ey'].to_numpy())))
        
    elast_prior = pd.DataFrame(np.reshape(prior_Es, (500, -1)), columns=e_labels)# .reindex(columns=elast_posterior.columns)

    fig = plt.figure(figsize=(20, 16))
    ax = fig.add_subplot(111)

    _ = sns.boxplot(data=elast_posterior, fliersize=0, ax=ax, zorder=2)
    
    prior_c = '0.7'
    _ = sns.boxplot(data=elast_prior, fliersize=0, zorder=0, showmeans=False,
                    capprops=dict(color=prior_c, zorder=0), medianprops=dict(color=prior_c, zorder=0.5),
                    whiskerprops=dict(color=prior_c, zorder=0), boxprops=dict(color=prior_c, facecolor='w', zorder=0), ax=ax)

    _ = ax.set_xticklabels(ax.get_xticklabels(), rotation=90)
    ax.axhline(0, ls='--', color='.5', zorder=1)
    ax.axvline(elast_nonzero.shape[1] - .5, color='.5', ls='--')

    ax.set_ylabel('Elasticity')
    sns.despine(trim=True)

    ax.set_yticks([-3, -2, -1, 0, 1, 2, 3])
    ax.set_ylim([-3, 3])

    ax.set_xlim(-.75, elast_nonzero.shape[1] + elast_zero.shape[1] - .5)
    sns.despine(ax=ax, trim=True)

    for tick in ax.xaxis.get_major_ticks():
        tick.label.set_fontsize(18) 

    plt.tight_layout()
    plt.title(f'{dataset_name} Predicted Elasticities')
    plt.savefig(results_dir + 'elast-plot/' + f'{dataset_name}-plotted_elasticities.svg', transparent=True)


def ADVI_CCs_hdi(trace, trace_prior, cc_type, r, ll, results_dir, cc_indexer=None):
    """
    Ex_hdi is the hdi of the posterior Ex trace as a numpy array
    """
    Ex_hdi = az.hdi(trace['posterior']['Ex'])['Ex'].to_numpy() #(13, 8, 2)
    Ey_hdi = az.hdi(trace['posterior']['Ey'])['Ey'].to_numpy() #(13, 2, 2)

    priorEx_hdi = az.hdi(trace_prior['prior']['Ex'])['Ex'].to_numpy() #(13, 8, 2)Ex_hdi = az.hdi(trace['posterior']['Ex'])['Ex'].to_numpy() #(13, 8, 2)
    
    a = np.transpose(Ex_hdi,(2, 0, 1))
    b = np.transpose(priorEx_hdi,(2, 0, 1))

    if cc_type=='mcc':
        cc_mb = np.array([ll.metabolite_control_coefficient(Ex=ex) for ex in a])   
        cc_prior = np.array([ll.metabolite_control_coefficient(Ex=ex) for ex in b]) 
        gt_ccs = calculate_gt_MCCs(r, cc_indexer) 
    elif cc_type=='fcc':
        cc_mb = np.array([ll.flux_control_coefficient(Ex=ex) for ex in a])   
        cc_prior = np.array([ll.flux_control_coefficient(Ex=ex) for ex in b]) 
        gt_ccs = calculate_gt_FCCs(r, cc_indexer)
    else: 
        raise Exception("cc_type must either be 'mcc' or 'fcc'")
    
    enzymesList = [i for i in r.getGlobalParameterIds() if 'E' in i]
    df2 = pd.DataFrame(cc_mb[:, 0], columns=enzymesList
                    ).stack().reset_index(level=1)
    df3 = pd.DataFrame(cc_prior[:, 0], columns=enzymesList
                    ).stack().reset_index(level=1)
    df2['Type'] = 'ADVI'
    df3['Type'] = 'Prior'

    cc_df = pd.concat([df2, df3])
    cc_df.columns = ['Enzymes', cc_type, 'Type']

    cc_hdi = pd.pivot_table(cc_df.reset_index(), values=cc_type, index =['Type', 'index'], columns='Enzymes')

    column_order = enzymesList
    cc_hdi = cc_hdi.reindex(column_order, axis=1)
    cc_hdi.to_csv(results_dir)

    medians = []
    for e in enzymesList:
        vals = df2[df2['level_1'] == e]
        vals.columns = ['_', 'val', '__']
        medians.append(vals['val'].median())

    with open(results_dir, 'a', newline='') as f:
        writer(f).writerow(['median',''] + medians)
        writer(f).writerow(['ground truth',''] + gt_ccs.tolist())
        f.close()

    return cc_df
    
    
def plot_CC_distbs(cc_df, cc_type, r, results_dir, dataset_name, cc_indexer=None):
    """
    objective: 
    Parameters---
    trace: 
    trace_prior: 
    ll: emll linlog object
    r: roadrunner object of model
    model: cobrapy object of model
    results_dir: filepath str for where result should go
    dataset_name: str description of filepath
    Return nothing. 
    """
    if cc_type=='mcc':
        gt_ccs = calculate_gt_MCCs(r, cc_indexer) 
        target_metabolite = cc_indexer
    elif cc_type=='fcc':
        gt_ccs = calculate_gt_FCCs(r, cc_indexer) 
        target_metabolite = r.getReactionIds()[-1]
    else: 
        raise Exception("cc_type must either be 'mcc' or 'fcc'")
    
    fig = plt.figure(figsize=(16, 8))

    my_pal = {"Prior": ".8", "ADVI":"b"}

    ax = fig.add_subplot(111)
    ax2 = fig.add_subplot(111, frameon=False, sharex=ax, sharey=ax)

    sns.violinplot(
        x='Enzymes', y=cc_type, hue='Type', data=cc_df[cc_df.Type == 'Prior'],
        scale='width', width=0.5, legend=False, zorder=0,
        color='1.', ax=ax, saturation=1., alpha=0.01)

    plt.setp(ax.lines, color='.8')
    plt.setp(ax.collections, alpha=.5, label="")

    sns.violinplot(
        x='Enzymes', y=cc_type, hue='Type', data=cc_df,
        scale='width', width=0.8, hue_order=['ADVI'],
        legend=False, palette=my_pal, zorder=3, ax=ax2)

    for i, cc in enumerate(gt_ccs):
        l = plt.plot([i - .4, i + .4], [cc, cc], '-', color=sns.color_palette('muted')[3])

    phandles, plabels = ax.get_legend_handles_labels()
    handles, labels = ax2.get_legend_handles_labels()
    ax.legend().remove()
    ax2.legend().remove()

    ax2.legend(phandles + handles + l, plabels + labels + ['Ground Truth'], loc='upper center', ncol=4, fontsize=13)
    ax.set_ylim([-1.5, 1.5])

    ax.axhline(0, ls='--', color='.7', zorder=0)
    sns.despine(trim=True)

    plt.suptitle(dataset_name + f' Predicted {cc_type.upper()}s: {target_metabolite}', y=1)

    fig.savefig(results_dir + f'{cc_type.upper()}-graph/{dataset_name}-plotted_{cc_type}s.svg', transparent=True)



