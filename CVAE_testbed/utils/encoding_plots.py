import argparse
import logging
from pathlib import Path
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
import pandas as pd
import seaborn as sns
from brokenaxes import brokenaxes
from mpl_toolkits.mplot3d import Axes3D
from CVAE_testbed.utils import str_to_object

LOGGER = logging.getLogger(__name__)

def make_plot_encoding(
        args: argparse.Namespace, model, df: pd.DataFrame
        ) -> None:
    sns.set_context("talk")
    path_save_dir = Path(args.path_save_dir)
    vis_enc = str_to_object(
        "CVAE_testbed.metrics.visualize_encoder.visualize_encoder_synthetic"
    )
    try:
        conds = [i for i in range(args.model_kwargs["dec_layers"][-1][-1])]
    except:
        conds = [i for i in range(args.model_kwargs["dec_layers"][-1])]
    
    try:
        latent_dims = args.model_kwargs["vae_layers"][-1][-1]
    except:
        latent_dims = args.model_kwargs["enc_layers"][-1]

    fig, ax = plt.subplots(1,1,figsize = (7, 5))
    sns.lineplot(ax=ax, data = df, x="epoch", y="total_train_ELBO")
    sns.lineplot(ax=ax, data = df, x="epoch", y="total_test_ELBO")
    ax.set_ylim([0, df.total_test_ELBO.quantile(0.95)])
    ax.legend(["Train loss", "Test loss"])
    ax.set_title("ELBO vs epoch")
    path_save_fig = path_save_dir / Path("ELBO.png")
    fig.savefig(path_save_fig, bbox_inches="tight")
    LOGGER.info(f"Saved: {path_save_fig}")

    fig, (ax1, ax, ax2, ax3) = plt.subplots(1, 4, figsize=(7 * 4, 5))
    fig2 = plt.figure(figsize=(12, 10))
    bax = brokenaxes(xlims=((0, latent_dims-50), (latent_dims - 4, latent_dims)), hspace=0.15)

    if "total_train_losses" in df.columns:
        sns.lineplot(ax=ax1, data=df, x="epoch", y="total_train_losses")
        ax1.set_xlabel("Epoch")
        ax1.set_ylabel("Loss")
    if "total_test_losses" in df.columns:
        sns.lineplot(ax=ax1, data=df, x="epoch", y="total_test_losses")
        ax1.set_xlabel("Epoch")
        ax1.set_ylabel("Loss")
        ax1.set_ylim([0, df.total_test_losses.quantile(0.95)])
        ax1.legend(["Train loss", "Test loss"])
    ax1.set_title("Loss vs epoch")

    try:
        this_kwargs = args.model_kwargs["dec_layers"][-1][-1]
    except:
        this_kwargs = args.model_kwargs["dec_layers"][-1]
    make_data = str_to_object(args.dataloader)
    this_dataloader = make_data(
        1, args.batch_size * 10, args.model_kwargs, shuffle=False
    )
    c, d, _, _ = this_dataloader.get_all_items()
    print('inside encoding', c[0,:].size(), d[0,:].size())

    conds = [i for i in range(this_kwargs)]
    if len(conds) > 30:
        conds = [i for i in conds if i%30 == 0]

    if args.post_plot_kwargs["latent_space_colorbar"] == "yes":
        color = this_dataloader.get_color()
    else:
        color = None

    for i in range(len(conds) + 1):
        print('inside main plot encoding', i, len(conds) + 1)
        if i == 0:
            z_means_x, z_means_y, kl_per_lt, _, _ = vis_enc(
                args,
                model,
                conds,
                c[0, :].clone(),
                d[0, :].clone(),
                kl_per_lt=None
            )
            ax.scatter(
                z_means_x,
                z_means_y,
                marker=".",
                s=30,
                label=str(i)
            )
            if color is not None:
                colormap_plot(
                    path_save_dir,
                    c, z_means_x,
                    z_means_y, color,
                    conds)
        else:
            z_means_x, z_means_y, kl_per_lt, _, _ = vis_enc(
                args,
                model,
                conds,
                c[0, :].clone(),
                d[0, :].clone(),
                kl_per_lt
            )
            ax.scatter(
                z_means_x,
                z_means_y,
                marker=".",
                s=30,
                label=str(i)
            )
            if color is not None:
                colormap_plot(
                    path_save_dir,
                    c,
                    z_means_x,
                    z_means_y,
                    color,
                    conds
                    )
        try:
            conds.pop()
        except:
            pass
    try:
        path_csv = path_save_dir / Path("visualize_encoding.csv")
        kl_per_lt = pd.DataFrame(kl_per_lt)
        kl_per_lt.to_csv(path_csv)
        LOGGER.info(f"Saved: {path_csv}")
    except:
        pass
    ax.set_title("Latent space")
    ax.legend()

    conds = [i for i in range(this_kwargs)]
    if len(conds) > 30:
        conds = [i for i in conds if i%30 == 0]

    for i in range(len(conds) + 1):
        tmp = kl_per_lt.loc[kl_per_lt["num_conds"] == c.size()[-1] - len(conds)]
        tmp = tmp.sort_values(
            by="kl_divergence",
            ascending=False
            )
        tmp = tmp.reset_index(drop=True)
        x = tmp.index.values
        y = tmp.iloc[:, 1].values
        sns.lineplot(
            ax=ax2,
            data=tmp,
            x=tmp.index,
            y="kl_divergence",
            label=str(i),
            legend='brief'
            )
        bax.plot(x, y, label=str(i))
        ax3.scatter(tmp['rcl'].mean(), tmp['kl_divergence'].mean(), label=str(i))
        # sns.scatterplot(
        #     ax=ax3,
        #     data=tmp,
        #     x="rcl",
        #     y="kl_divergence",
        #     label=str(i),
        #     legend='brief'
        #     )
        try:
            conds.pop()
        except:
            pass

    ax2.set_xlabel("Latent dimension")
    ax2.set_ylabel("KLD")
    ax2.set_title("KLD per latent dim")
    ax3.set_xlabel("MSE")
    ax3.set_ylabel("KLD")
    ax3.set_title("MSE vs KLD")
    bax.legend(loc="best")
    bax.set_xlabel("Latent dimension")
    bax.set_ylabel("KLD")
    bax.set_title("KLD per latent dim")

    try:
        path_save_fig = path_save_dir / Path("encoding_test_plots.png")
        fig.savefig(path_save_fig, bbox_inches="tight")
        LOGGER.info(f"Saved: {path_save_fig}")
    except:
        pass

    try:
        path_save_fig = path_save_dir / Path("brokenaxes_KLD_per_dim.png")
        fig2.savefig(path_save_fig, bbox_inches="tight")
        LOGGER.info(f"Saved: {path_save_fig}")
    except:
        pass


def colormap_plot(
    path_save_dir, swiss_roll, z_means_x, z_means_y, color, conds
) -> None:

    # fig, ax = plt.subplots(1, 2, figsize=(7*2,5))
    fig = plt.figure(figsize=(7 * 2, 5))

    ax1 = fig.add_subplot(121, projection="3d")
    swiss_roll = swiss_roll[0, :]
    try:
        ax1.scatter(swiss_roll[:, 0], swiss_roll[:, 1], swiss_roll[:, 2], c=color)
    except:
        ax1.scatter(swiss_roll[:, 0], swiss_roll[:, 1], c=color)

    ax = fig.add_subplot(122)
    divider = make_axes_locatable(ax)
    cax = divider.append_axes(
        "right", size="5%",
        pad=0.05,
        title="arc length"
        )
    a = ax.scatter(z_means_x, z_means_y, c=color)
    fig.colorbar(a, cax=cax)

    path_save_fig = path_save_dir / Path(
        "latent_space_colormap_conds_" + str(3 - len(conds)) + ".png"
    )
    fig.savefig(path_save_fig, bbox_inches="tight")
    LOGGER.info(f"Saved: {path_save_fig}")