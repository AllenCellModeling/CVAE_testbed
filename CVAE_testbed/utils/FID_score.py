import argparse
import logging
from pathlib import Path
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import numpy as np
import torch
import math
from CVAE_testbed.utils import str_to_object

LOGGER = logging.getLogger(__name__)


def make_plot_FID(args: argparse.Namespace, model, X_test, C_test, save=True):

    X_test = X_test.view(-1, X_test.size()[-1])
    C_test = C_test.view(-1, C_test.size()[-1])
    # X_test = X_test[-1,:]
    # C_test = C_test[-1,:]
    print(X_test.size(), C_test.size())

    sns.set_context("talk")
    path_save_dir = Path(args.path_save_dir)
    compute_fid = str_to_object(
        "CVAE_testbed.run_models.generative_metric.compute_generative_metric_synthetic"
    )

    try:
        this_kwargs = args.model_kwargs["dec_layers"][-1][-1]
    except:
        this_kwargs = args.model_kwargs["dec_layers"][-1]

    # ADD YOUR PATH HERE
    csv_greedy_features = pd.read_csv('~/Github/cookiecutter/CVAE_testbed/scripts' + args.path_save_dir[1:] + '/selected_features.csv')

    #conds = [i for i in range(this_kwargs)]
    conds = [i for i in csv_greedy_features['selected_feature_number'] if not math.isnan(i)]


    fid_data = {'num_conds': [], 'fid': []}

    print(conds)

    for i in range(len(conds) + 1):

        tmp1, tmp2 = torch.split(
            C_test.clone(),
            int(C_test.clone().size()[-1] / 2),
            dim=1
                                )
        for kk in conds:
            tmp1[:, int(kk)], tmp2[:, int(kk)] = 0, 0
        cond_d = torch.cat((tmp1, tmp2), 1)

        print(len(torch.nonzero(cond_d)))


        try:
            this_fid = compute_fid(X_test.clone(), cond_d.clone(), args, model, conds)
        except:
            this_fid = np.NaN
        print('fid', this_fid)
                
        fid_data['num_conds'].append(X_test.size()[-1] - len(conds))
        fid_data['fid'].append(this_fid)

        try:
            conds.pop()
        except:
            pass

    fid_data = pd.DataFrame(fid_data)

    fig, ax = plt.subplots(1, 1, figsize=(7*4, 5))
    sns.lineplot(ax=ax, data=fid_data, x='num_conds', y='fid')
    sns.scatterplot(ax=ax, data=fid_data, x='num_conds', y='fid', s=100, color=".2")

    if save is True:
        path_csv = path_save_dir / Path("fid_data.csv")
        fid_data.to_csv(path_csv)
        LOGGER.info(f"Saved: {path_csv}")

        path_save_fig = path_save_dir / Path("fid_score.png")
        fig.savefig(path_save_fig, bbox_inches="tight")
        LOGGER.info(f"Saved: {path_save_fig}")
    

