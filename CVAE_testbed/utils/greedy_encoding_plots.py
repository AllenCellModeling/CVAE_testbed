import argparse
import logging
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from CVAE_testbed.utils import str_to_object

LOGGER = logging.getLogger(__name__)


def make_plot_encoding_greedy(args: argparse.Namespace, model, df: pd.DataFrame
        ) -> None:
    sns.set_context("talk")
    path_save_dir = Path(args.path_save_dir)
    vis_enc = str_to_object(
        "CVAE_testbed.metrics.greedy_visualize_encoder.GreedyVisualizeEncoder"
    )
    try:
        conds = [i for i in range(args.model_kwargs["dec_layers"][-1][-1])]
    except:
        conds = [i for i in range(args.model_kwargs["dec_layers"][-1])]

    make_data = str_to_object(args.dataloader)
    this_dataloader = make_data(
        1, args.batch_size * 10, args.model_kwargs, shuffle=False
    )
    c, d, _, _ = this_dataloader.get_all_items()

    kl_per_lt, kl_all_lt = vis_enc(
        args,
        model,
        conds,
        c[0, :].clone(),
        d[0, :].clone(),
        kl_per_lt=None,
        kl_all_lt=None
    )


    kl_per_lt, kl_all_lt = pd.DataFrame(kl_per_lt), pd.DataFrame(kl_all_lt)

    fig, (ax, ax1, ax2) = plt.subplots(1, 3, figsize=(7 * 3, 4))
    sns.lineplot(ax=ax, data=kl_per_lt, x = 'num_conds', y = 'elbo', estimator='mean')
    sns.lineplot(ax=ax1, data=kl_per_lt, x = 'num_conds', y = 'kl_divergence', estimator='mean')
    sns.lineplot(ax=ax2, data=kl_per_lt, x = 'num_conds', y = 'rcl', estimator='mean')
    path_save_fig = path_save_dir / Path("greedy_elbo_kld_rcl_conds.png")
    fig.savefig(path_save_fig, bbox_inches="tight")
    LOGGER.info(f"Saved: {path_save_fig}")

    fig, (ax, ax1, ax2) = plt.subplots(1, 3, figsize=(7 * 3, 4))
    sns.lineplot(ax=ax, data=kl_per_lt, x = 'latent_dim', y = 'elbo', estimator='mean')
    sns.lineplot(ax=ax1, data=kl_per_lt, x = 'latent_dim', y = 'kl_divergence', estimator='mean')
    sns.lineplot(ax=ax2, data=kl_per_lt, x = 'latent_dim', y = 'rcl', estimator='mean')

    path_save_fig = path_save_dir / Path("greedy_elbo_kld_rcl_dims.png")
    fig.savefig(path_save_fig, bbox_inches="tight")
    LOGGER.info(f"Saved: {path_save_fig}")

    kld_per_dim = pd.pivot_table(kl_per_lt, values='kl_divergence', index='num_conds', columns = 'cond_comb', aggfunc=np.sum)

    rcl_per_dim = pd.pivot_table(kl_per_lt, values='rcl', index='num_conds', columns = 'cond_comb', aggfunc=np.sum)

    elbo_per_dim = pd.pivot_table(kl_per_lt, values='elbo', index='num_conds', columns = 'cond_comb', aggfunc=np.sum)

    fig2, (ax, ax2, ax3) = plt.subplots(1, 3, figsize=(10 * 3, 5))
    sns.heatmap(elbo_per_dim, ax=ax, cmap = 'coolwarm')
    ax.set_title('-ELBO')

    sns.heatmap(kld_per_dim, ax=ax2, cmap = 'coolwarm')
    ax2.set_title('KLD')
    sns.heatmap(rcl_per_dim, ax=ax3, cmap = 'coolwarm')
    ax3.set_title('RCL')

    path_save_fig = path_save_dir / Path("greedy_heatmaps.png")
    fig2.savefig(path_save_fig, bbox_inches="tight")
    LOGGER.info(f"Saved: {path_save_fig}")

    df = kl_all_lt.loc[kl_all_lt['num_conds'] == c.size()[-1] - 1]
    this_z_means_x = df['z_means_x'].values
    this_z_means_y = df['z_means_y'].values

    fig4, ax = plt.subplots(1,1,figsize = (6, 5))
    ax.scatter(this_z_means_x[0], this_z_means_y[0], marker=".", s=30, label= str(0))

    print('greedy', c.size()[-1])
    df = kl_all_lt.loc[kl_all_lt['num_conds'] == c.size()[-1]]

    this_z_means_x = df['z_means_x'].values
    this_z_means_y = df['z_means_y'].values
    ax.scatter(this_z_means_x[0], this_z_means_y[0], marker=".", s=30, label= str(c.size()[-1]))
    ax.set_title("Latent space")
    ax.legend()
    path_save_fig = path_save_dir / Path("encoding_latent_space.png")
    fig4.savefig(path_save_fig, bbox_inches="tight")
    LOGGER.info(f"Saved: {path_save_fig}")