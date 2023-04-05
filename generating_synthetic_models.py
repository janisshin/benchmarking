import SBbadger
import numpy as np
import tellurium as te
import teUtils as tu
import csv
import re
from tqdm import tqdm

def in_dist(k):
    return 159.56679886918792*k**(-1.6914416803009278)

def out_dist(k):
    return 152.91017262603404*k**(-1.2471437815467692)

if __name__ == "__main__":

    SBbadger.generate_serial.models(
        group_name='mass_action',
        n_models=1000,
        n_species=10,
        out_dist=out_dist,
        in_dist=in_dist,
        add_enzyme=True,
        rxn_prob=[.35, .30, .30, .05],
        kinetics=['mass_action', ['loguniform', 'loguniform', 'loguniform'],
                  ['kf', 'kr', 'kc'],
                  [[0.01, 100], [0.01, 100], [0.01, 100]]],
        allo_reg=[[0.5, 0.5, 0, 0], 0.5, ['uniform', 'uniform', 'uniform'],
                  ['ro', 'kma', 'ma'],
                  [[0, 1], [0, 1], [0, 1]]],
        overwrite=True,
        constants=False,
        source=[1, 'loguniform', 0.01, 1, 1],
        sink=[2, 'loguniform', 0.01, 1],
        rev_prob=1,
        ic_params=['uniform', 0, 10],
        net_plots=True,
        cobra=True
    )