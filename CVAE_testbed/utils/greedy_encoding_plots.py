import argparse
import logging
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from CVAE_testbed.utils import str_to_object

LOGGER = logging.getLogger(__name__)


def make_plot_encoding_greedy(
                                args: argparse.Namespace,
                                model, df: pd.DataFrame,
                                c,
                                d,
                                feature_names=None,
                                save=True,
                                proj_matrix=None
                             ) -> None:
    """
    c and d are X_test and C_test
    """
    sns.set_context("talk")
    path_save_dir = Path(args.path_save_dir)
    vis_enc = str_to_object(
        "CVAE_testbed.metrics.greedy_visualize_encoder.GreedyVisualizeEncoder"
    )
    try:
        conds = [i for i in range(args.model_kwargs["dec_layers"][-1][-1])]
    except:
        conds = [i for i in range(args.model_kwargs["dec_layers"][-1])]

    kl_per_lt, kl_all_lt, selected_features, first_features = vis_enc(
        args,
        model,
        conds,
        c[-1, :].clone(),
        d[-1, :].clone(),
        kl_per_lt=None,
        kl_all_lt=None,
        selected_features=None,
        feature_names=feature_names
    )

    kl_per_lt, kl_all_lt, selected_features, first_features = pd.DataFrame(kl_per_lt), pd.DataFrame(kl_all_lt), pd.DataFrame(selected_features), pd.DataFrame(first_features)

    if save is True:
        path_csv = path_save_dir / Path("kl_per_lt.csv")
        kl_per_lt.to_csv(path_csv)
        LOGGER.info(f"Saved: {path_csv}")

        path_csv = path_save_dir / Path("kl_all_lt.csv")
        kl_all_lt.to_csv(path_csv)
        LOGGER.info(f"Saved: {path_csv}")

        path_csv = path_save_dir / Path("selected_features.csv")
        selected_features.to_csv(path_csv)
        LOGGER.info(f"Saved: {path_csv}")

        path_csv = path_save_dir / Path("first_features.csv")
        first_features.to_csv(path_csv)
        LOGGER.info(f"Saved: {path_csv}")

    fig, ax1 = plt.subplots(1, 1, figsize=(7 * 1, 4))
    sns.lineplot(ax=ax1, data=kl_per_lt, x='latent_dim', y='kl_divergence', estimator='mean')

    if save is True:
        path_save_fig = path_save_dir / Path("greedy_elbo_kld_rcl_dims.png")
        fig.savefig(path_save_fig, bbox_inches="tight")
        LOGGER.info(f"Saved: {path_save_fig}")

    fig2, ax = plt.subplots(1, 1, figsize=(7 * 8, 4))
    first_features.sort_values(by='ELBO', ascending=False, inplace=True)

    if all(pd.isna(first_features['selected_feature_name'])):
        bar_fig = sns.lineplot(data=first_features, ax=ax, x='selected_feature_number', y='ELBO', label='ELBO', sort=False)
        sns.scatterplot(data=first_features, ax=ax, x='selected_feature_number', y='ELBO', s=100, color=".2")
        sns.lineplot(data=first_features, ax=ax, x='selected_feature_number', y='RCL', label='RCL', sort=False)
        sns.scatterplot(data=first_features, ax=ax, x='selected_feature_number', y='RCL', s=100, color=".2")
    else:
        bar_fig = sns.lineplot(data=first_features, ax=ax, x='selected_feature_name', y='ELBO', label="ELBO", sort=False)
        sns.scatterplot(data=first_features, ax=ax, x='selected_feature_name', y='ELBO', s=100, color=".2")
        sns.lineplot(data=first_features, ax=ax, x='selected_feature_name', y='RCL', label="RCL", sort=False)
        sns.scatterplot(data=first_features, ax=ax, x='selected_feature_name', y='RCL', s=100, color=".2")

    for item in bar_fig.get_xticklabels():
        item.set_rotation(45)

    ax.set_title('ELBO per selected first feature')  
    ax.set_xlabel('Selected feature')
    ax.set_ylabel('ELBO')

    if save is True:
        path_save_fig = path_save_dir / Path("greedy_barplots_first_selection.png")
        fig2.savefig(path_save_fig, bbox_inches="tight")
        LOGGER.info(f"Saved: {path_save_fig}")

    fig2, ax = plt.subplots(1, 1, figsize=(7 * 8, 4))

    print(selected_features)

    selected_features.sort_values(by='ELBO', ascending=False, inplace=True)

    if all(pd.isna(selected_features['selected_feature_name'])):
        bar_fig = sns.lineplot(data=selected_features, ax=ax, x='selected_feature_number', y='ELBO', label='ELBO', sort=False)
        sns.scatterplot(data=selected_features, ax=ax, x='selected_feature_number', y='ELBO', s=100, color=".2")
        sns.lineplot(data=selected_features, ax=ax, x='selected_feature_number', y='RCL', label='RCL',sort=False)
        sns.scatterplot(data=selected_features, ax=ax, x='selected_feature_number', y='RCL', s=100, color=".2")
    else:
        bar_fig = sns.lineplot(data=selected_features, ax=ax, x='selected_feature_name', y='ELBO', label='ELBO', sort=False)
        sns.scatterplot(data=selected_features, ax=ax, x='selected_feature_name', y='ELBO', s=100, color=".2")
        sns.lineplot(data=selected_features, ax=ax, x='selected_feature_name', y='RCL', label='RCL',sort=False)
        sns.scatterplot(data=selected_features, ax=ax, x='selected_feature_name', y='RCL', s=100, color=".2")

    for item in bar_fig.get_xticklabels():
        item.set_rotation(45)

    ax.set_title('ELBO per selected feature')
    ax.set_xlabel('Selected feature')
    ax.set_ylabel('ELBO')
    if save is True:
        path_save_fig = path_save_dir / Path("greedy_barplots.png")
        fig2.savefig(path_save_fig, bbox_inches="tight")
        LOGGER.info(f"Saved: {path_save_fig}")