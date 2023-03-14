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
pd.set_option
('display.max_columns', None)
pd.set_option('display.max_rows', None)

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

# linlog Bayesian Metabolic Control Analysis
import emll
from emll.util import initialize_elasticity

def locateData(passlist, folder_name='./'):
    """
    passlist is a literal list of filenames of models that passed the filter criteria

    what is b?? name of a filename?
        if "handle" is in the name of the generated data file, 
        make sure to collect that data

    you're trying to make a list of lists
    The inner lists will have different dataset types for one type of model
    the larger list will have a collections of datasets from all passed models.
    """
    datasets = []

    for passN in passlist:
        modelset = []
        for dataset_name in os.listdir(folder_name + 'generated_data/'):
            if f'data_{passN}_' in dataset_name:
                modelset.append(dataset_name)
        datasets.append(modelset)
    return datasets


def run_analysis(path, dataPath, itr=30000, folder_name="./"):
    
    r = te.loads(folder_name + 'sbml/' + path)
    model = cobra.io.read_sbml_model(folder_name + 'sbml/' + path)

    ex_list = [i.id for i in list(model.reactions) if 'EX' in i.id]
    model.remove_reactions(ex_list)

    removable_mets = [i for i in list(model.metabolites) if 'B' in i.id]
    # if removable_mets in a reaction, write down the name of the reaction
    # this code assumes that there will only be one source and one sink
    ex_sp_rxns = [list(model.metabolites.get_by_id(m.id).reactions)[0].id for m in removable_mets]
    for met in removable_mets:
        model.remove_metabolites(met)
    
    # return r, model
    data = pd.read_csv(folder_name + 'generated_data/' + dataPath).astype(float)

    N = r.getFullStoichiometryMatrix()
    nm, nr = N.shape
    
    n_exp = len(data)   

    e_cols = [col for col in data.columns if 'E' in col]
    x_cols = r.getFloatingSpeciesIds()    
    y_cols = [col for col in r.getBoundarySpeciesIds()]
    v_cols = [col for col in data.columns if 'flux' in col]

    e = data[e_cols]
    x = data[x_cols]
    y = data[y_cols]
    v = data[v_cols]

    ref_ind = 0 ## corresponds to how the data was generated

    e_star = e.iloc[ref_ind].values
    x_star = x.iloc[ref_ind].values
    y_star = y.iloc[ref_ind].values
    v_star = v.iloc[ref_ind].values

    v_star[v_star == 0] = 1e-9
    y_star[y_star == 0] = 1e-6

    # Normalize to reference values (and drop trivial measurement)
    en = e.values / e_star
    xn = x.values / x_star
    yn = y.values / y_star
    vn = v.values / v_star[ref_ind]

    N[:, v_star < 0] = -1 * N[:, v_star < 0]
    v_star = np.abs(v_star)

    # for some reason, there are 21 reactions in the model
    # there should only be 19
    Ex = emll.create_elasticity_matrix(model)

    Ey = np.zeros((nr, len(y_cols))) # (reactions, number of external species)
    
    # external species reaction number--the reaction number where the external species appears? (list)
    ex_sp_rxn_ns = [i[1:] for i in ex_sp_rxns]

    # for each external species:    
    for i in range(len(removable_mets)): 
        Ey[int(ex_sp_rxn_ns[i]), i] = 1

    ll = emll.LinLogLeastNorm(N, Ex, Ey, v_star)

    ### creating the probability model
    with pm.Model() as pymc_model:
        
        # Initialize elasticities
        Ex_t = pm.Deterministic('Ex', initialize_elasticity(N, 'ex', b=0.05, sd=1, alpha=5))
        Ey_t = pm.Deterministic('Ey', initialize_elasticity(-Ey.T, 'ey', b=0.05, sd=1, alpha=5))
            
    with pymc_model:
        
        # Error priors. 
        v_err = pm.HalfNormal('v_error', sigma=0.05, initval=.1)
        x_err = pm.HalfNormal('x_error', sigma=0.05, initval=.1) # shape must match so that pm.Normal() runs successfully

        # Calculate steady-state concentrations and fluxes from elasticities
        chi_ss, v_hat_ss = ll.steady_state_aesara(Ex_t, Ey_t, en, yn)

        # Error distributions for observed steady-state concentrations and fluxes
        
        v_hat_obs = pm.Normal('v_hat_obs', mu=v_hat_ss, sigma=v_err, observed=vn) # both bn and v_hat_ss are (28,6)
        chi_obs = pm.Normal('chi_obs', mu=chi_ss,  sigma=x_err,  observed=xn) # chi_ss and xn is (28,4)

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
    y_labels = removable_mets

    ex_labels = np.array([['$\epsilon_{' + '{0},{1}'.format(rlabel, mlabel) + '}$'
                        for mlabel in m_labels] for rlabel in r_labels]).flatten()
    ey_labels = np.array([['$\epsilon_{' + '{0},{1}'.format(rlabel, mlabel) + '}$'
                        for mlabel in y_labels] for rlabel in r_labels]).flatten()

    e_labels = np.hstack((ex_labels, ey_labels))
    
    # generating and storing results
    results_dir = './' + folder_name + 'results/'
    dataset_name = dataPath.split(".")[0]

    #plot_ADVI_converg(approx, itr, results_dir + 'convergence/', dataset_name)
    
    priorEx, Ex_hdi, Ey_hdi = calculate_elasticities_hdi(trace, trace_prior)
    #elasticities_to_csv(Ex_hdi, Ey_hdi, e_labels, results_dir + 'elast-hpd/' + dataset_name)
    #plot_elasticities(trace, trace_prior, N, e_labels, results_dir, dataset_name)
    #analyze_ADVI_MCCs(trace, trace_prior, ll, r, model, results_dir, dataset_name)
    #analyze_ADVI_FCCs(trace, trace_prior, ll, r, model, results_dir, dataset_name)

def calculate_gt_FCCs(r):
    """
    objective: directly calculates the FCCs of model
    Parameters
    r: roadrunner object of model
    Returns FCCs for sink reaction
    """
    r.steadyState()
    return r.getScaledFluxControlCoefficientMatrix()[-1]


def calculate_gt_MCCs(r):
    """
    objective: directly calculates the CCCs of model
    Parameters
    r: roadrunner object of model
    Returns CCCs for sink reaction
    """
    target = None
    for sp in r.getBoundarySpeciesIds():
        if r.getValue(sp) == 0:
            target = 'S'+str(sp[1:])
    r.steadyState()
    return r.getScaledConcentrationControlCoefficientMatrix()[target]


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


def calculate_elasticities_hdi(trace, trace_prior):

    Ex_hdi = az.hdi(trace['posterior']['Ex'])['Ex'].to_numpy() #(13, 8, 2)
    Ey_hdi = az.hdi(trace['posterior']['Ey'])['Ey'].to_numpy() #(13, 2, 2)

    priorEx_hdi = az.hdi(trace_prior['prior']['Ex'])['Ex'].to_numpy() #(13, 8, 2)

    return Ex_hdi, Ey_hdi, priorEx_hdi


def elasticities_to_csv(Ex_hdi, Ey_hdi, e_labels, results_dir):

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


def ADVI_MCCs_hdi(Ex_hdi, priorEx_hdi):
    """
    Ex_hdi is the hdi of the posterior Ex trace as a numpy array
    """
    a = np.transpose(Ex_hdi,(2, 0, 1))
    b = np.transpose(priorEx_hdi,(2, 0, 1))
    mcc_mb = np.array([ll.metabolite_control_coefficient(Ex=ex) for ex in a])   
    mcc_prior = np.array([ll.metabolite_control_coefficient(Ex=ex) for ex in b]) 

    df2 = pd.DataFrame(mcc_mb[:, 0], columns=[r.id for r in model.reactions]
                  ).stack().reset_index(level=1)
    df3 = pd.DataFrame(mcc_prior[:, 0], columns=[r.id for r in model.reactions]
                    ).stack().reset_index(level=1)
    df2['type'] = 'ADVI'
    df3['type'] = 'Prior'

    mcc_df = pd.concat([df2, df3])
    mcc_df.columns = ['Reaction', 'mcc', 'Type']

    mcc_df.loc[mcc_df.mcc < -.5, 'mcc'] = np.nan
    mcc_df.loc[mcc_df.mcc > 1.5, 'mcc'] = np.nan
    medians = list()
    
    for e in r.getReactionIds(): # what is this line doing
        vals = df2[df2['level_1'] == e]
        vals.columns = ['_', 'val', '__']
        medians.append(vals['val'].median())
    median_df = pd.DataFrame(medians, index=r.getReactionIds())

def analyze_ADVI_MCCs(trace, trace_prior, ll, r, model, results_dir, dataset_name):
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
    priorEx = np.squeeze(trace_prior['prior']['Ex'].to_numpy()) # (500, 17, 17)
    postEx = np.squeeze(trace['posterior']['Ex'].to_numpy()) # (1000, 17, 17)
    # postEy = np.squeeze(trace['posterior']['Ey'].to_numpy()) # 

    
    mcc_mb = np.array([ll.metabolite_control_coefficient(Ex=ex) for ex in postEx])   
    mcc_prior = np.array([ll.metabolite_control_coefficient(Ex=ex) for ex in priorEx]) 
    
    df2 = pd.DataFrame(mcc_mb[:, 0], columns=[r.id for r in model.reactions]
                  ).stack().reset_index(level=1)
    df3 = pd.DataFrame(mcc_prior[:, 0], columns=[r.id for r in model.reactions]
                    ).stack().reset_index(level=1)

    df2['type'] = 'ADVI'
    df3['type'] = 'Prior'

    mcc_df = pd.concat([df2, df3])
    mcc_df.columns = ['Reaction', 'mcc', 'Type']

    mcc_df.loc[mcc_df.mcc < -.5, 'mcc'] = np.nan
    mcc_df.loc[mcc_df.mcc > 1.5, 'mcc'] = np.nan

    medians = list()
    
    for e in r.getReactionIds(): # what is this line doing
        vals = df2[df2['level_1'] == e]
        vals.columns = ['_', 'val', '__']
        medians.append(vals['val'].median())
    median_df = pd.DataFrame(medians, index=r.getReactionIds())
    median_df.to_csv(results_dir + 'mcc-median/' + f'{dataset_name}-median_MCCs.csv')

    # plot 
    fig = plt.figure(figsize=(16, 8))

    my_pal = {"Prior": ".8", "ADVI":"b"}

    ax = fig.add_subplot(111)
    ax2 = fig.add_subplot(111, frameon=False, sharex=ax, sharey=ax)

    sns.violinplot(
        x='Reaction', y='mcc', hue='Type', data=mcc_df[mcc_df.Type == 'Prior'],
        scale='width', width=0.5, legend=False, zorder=0,
        color='1.', ax=ax, saturation=1., alpha=0.01)

    plt.setp(ax.lines, color='.8')
    plt.setp(ax.collections, alpha=.5, label="")

    sns.violinplot(
        x='Reaction', y='mcc', hue='Type', data=mcc_df,
        scale='width', width=0.8, hue_order=['ADVI'],
        legend=False, palette=my_pal, zorder=3, ax=ax2)

    gt_mccs = calculate_gt_MCCs(r) 

    for i, cc in enumerate(gt_mccs):
        l = plt.plot([i - .4, i + .4], [cc, cc], '-', color=sns.color_palette('muted')[3])

    phandles, plabels = ax.get_legend_handles_labels()
    handles, labels = ax2.get_legend_handles_labels()
    ax.legend().remove()
    ax2.legend().remove()

    ax2.legend(phandles + handles + l, plabels + labels + ['Ground Truth'], loc='upper center', ncol=4, fontsize=13)
    ax.set_ylim([-1, 2])

    ax.axhline(0, ls='--', color='.7', zorder=0)
    sns.despine(trim=True)

    plt.suptitle(dataset_name + 'Predicted MCCs', y=1)

    fig.savefig(results_dir + 'mcc-graph/' + f'{dataset_name}-plotted_MCCs.svg', transparent=True)


def analyze_ADVI_FCCs(trace, trace_prior, ll, r, model, results_dir, dataset_name):
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
    priorEx = np.squeeze(trace_prior['prior']['Ex'].to_numpy()) # (500, 17, 17)
    postEx = np.squeeze(trace['posterior']['Ex'].to_numpy()) # (1000, 17, 17)
    # postEy = np.squeeze(trace['posterior']['Ey'].to_numpy()) # 

    
    fcc_mb = np.array([ll.flux_control_coefficient(Ex=ex) for ex in postEx])   
    fcc_prior = np.array([ll.flux_control_coefficient(Ex=ex) for ex in priorEx]) 
    
    df2 = pd.DataFrame(fcc_mb[:, 0], columns=[r.id for r in model.reactions]
                  ).stack().reset_index(level=1)
    df3 = pd.DataFrame(fcc_prior[:, 0], columns=[r.id for r in model.reactions]
                    ).stack().reset_index(level=1)

    df2['type'] = 'ADVI'
    df3['type'] = 'Prior'

    fcc_df = pd.concat([df2, df3])
    fcc_df.columns = ['Reaction', 'FCC', 'Type']

    fcc_df.loc[fcc_df.FCC < -.5, 'FCC'] = np.nan
    fcc_df.loc[fcc_df.FCC > 1.5, 'FCC'] = np.nan

    medians = list()
    
    for e in r.getReactionIds(): # what is this line doing
        vals = df2[df2['level_1'] == e]
        vals.columns = ['_', 'val', '__']
    medians.append(vals['val'].median())
    median_df = pd.DataFrame(medians)
    median_df.to_csv(results_dir + 'FCC-median/' + f'{dataset_name}-median_FCCs.csv')

    # plot 
    fig = plt.figure(figsize=(16, 8))

    my_pal = {"Prior": ".8", "ADVI":"b"}

    ax = fig.add_subplot(111)
    ax2 = fig.add_subplot(111, frameon=False, sharex=ax, sharey=ax)

    sns.violinplot(
        x='Reaction', y='FCC', hue='Type', data=fcc_df[fcc_df.Type == 'Prior'],
        scale='width', width=0.5, legend=False, zorder=0,
        color='1.', ax=ax, saturation=1., alpha=0.01)

    plt.setp(ax.lines, color='.8')
    plt.setp(ax.collections, alpha=.5, label="")

    sns.violinplot(
        x='Reaction', y='FCC', hue='Type', data=fcc_df,
        scale='width', width=0.8, hue_order=['ADVI'],
        legend=False, palette=my_pal, zorder=3, ax=ax2)

    gt_FCCs = calculate_gt_FCCs(r) 

    for i, cc in enumerate(gt_FCCs):
        l = plt.plot([i - .4, i + .4], [cc, cc], '-', color=sns.color_palette('muted')[3])

    phandles, plabels = ax.get_legend_handles_labels()
    handles, labels = ax2.get_legend_handles_labels()
    ax.legend().remove()
    ax2.legend().remove()

    ax2.legend(phandles + handles + l, plabels + labels + ['Ground Truth'], loc='upper center', ncol=4, fontsize=13)
    ax.set_ylim([-1, 2])

    ax.axhline(0, ls='--', color='.7', zorder=0)
    sns.despine(trim=True)

    plt.suptitle(dataset_name + 'Predicted FCCs', y=1)

    fig.savefig(results_dir + 'FCC-graph/' + f'{dataset_name}-plotted_FCCs.svg', transparent=True)



