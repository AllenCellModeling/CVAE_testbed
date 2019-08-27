import argparse
import logging
import json
import time
from pathlib import Path
import datetime
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable

import torch
from torch import nn, optim
from brokenaxes import brokenaxes

from typing import Optional, Dict
import pandas as pd
import numpy as np
from mpl_toolkits.mplot3d import Axes3D
from CVAE_testbed.models.model_loader import ModelLoader
from CVAE_testbed.utils import str_to_object

import seaborn as sns

LOGGER = logging.getLogger(__name__)


def get_args():
    """
    Get args from .sh
    """

    def load_synthetic(x):
        try:
            with open(x) as fp:
                a = json.load(fp)
            return a
        except:
            o = json.loads(x)
            return o

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--batch_size",
        type=int,
        default=5,
        help="Mini-batch size"
    )
    parser.add_argument(
        "--num_batches",
        type=int,
        default=10,
        help="Number of mini-batches"
    )
    parser.add_argument(
        "--beta_vae",
        type=int,
        default=1,
        help="Beta parameter in beta*(RCL + KLD) loss"
    )
    parser.add_argument(
        "--C_vae",
        type=int,
        default=0,
        help="C parameter in RCL + (KLD - C) loss"
    )
    parser.add_argument(
        "--dataloader",
        default="CVAE_testbed.datasets.synthetic.SyntheticDataset",
        help="Data set to load",
    )
    parser.add_argument("--gpu_id", default=0, type=int, help="GPU ID")
    parser.add_argument(
        "--loss_fn",
        default="CVAE_testbed.losses.ELBO.synthetic_loss",
        help="loss_function",
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=0.001,
        help="Learning rate")
    parser.add_argument(
        "--model_fn",
        default="CVAE_testbed.models.CVAE_baseline.CVAE",
        help=", Model name",
    )
    parser.add_argument(
        "--model_kwargs",
        type=lambda x: load_synthetic(x),
        default={
            "x_dim": 2,
            "c_dim": 4,
            "enc_layers": [2, 64, 64, 64, 64],
            "dec_layers": [64, 64, 64, 64, 2],
        },
        help="Model kwargs",
    )
    parser.add_argument(
        "--post_plot_kwargs",
        type=lambda x: load_synthetic(x),
        default={
            "latent_space_colorbar": False,
        },
        help="Post-hoc plot kwargs",
    )
    parser.add_argument(
        "--n_epochs", type=int, default=10, help="Number of training epochs"
    )
    parser.add_argument(
        "--data_type", default="synthetic", help="mnist or synthetic data"
    )
    parser.add_argument(
        "--path_save_dir",
        default="CVAE_testbed/outputs/baseline_results/",
        help="Model save directory",
    )
    args = parser.parse_args()

    args.path_save_dir = args.path_save_dir + str(
        datetime.datetime.today().strftime("%Y:%m:%d:%H:%M:%S")
    )
    return args


def save_args(
        args: argparse.Namespace,
        path_json: Optional[Path] = None
        ) -> None:
    """Save command-line arguments as json.s
    Parameters
    ----------
    args
        Command-line arguments.
    path_json
        JSON save path.
    Returns
    -------
    None
    """
    if path_json is None:
        path_json = Path(args.path_save_dir) / Path("arguments.json")
    with path_json.open("w") as fo:
        json.dump(vars(args), fo, indent=4)
    LOGGER.info(f"Saved: {path_json}")


def get_model(model_fn, model_kwargs: Optional[Dict] = None) -> nn.Module:
    model_fn = str_to_object(model_fn)
    try:
        return model_fn(**model_kwargs)
    except:
        a = dict(
            [
                (key, value)
                for key, value in model_kwargs.items()
                if key != "sklearn_data"
            ]
        )
        return model_fn(**a)


def colorbar(mappable):
    ax = mappable.axes
    fig = ax.figure
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.05)
    return fig.colorbar(mappable, cax=cax)


def train_model():
    """
    Trains a model
    """
    tic = time.time()
    args = get_args()

    path_save_dir = Path(args.path_save_dir)
    if path_save_dir.exists():
        raise ValueError(f"Save directory already exists! ({path_save_dir})")
    path_save_dir.mkdir(parents=True)

    logging.getLogger().addHandler(
        logging.FileHandler(path_save_dir / Path("run.log"), mode="w")
    )
    save_args(args, path_save_dir / Path("training_options.json"))

    device = (
        torch.device("cuda", args.gpu_id)
        if torch.cuda.is_available()
        else torch.device("cpu")
    )
    LOGGER.info(f"Using device: {device}")

    if args.data_type == "mnist":
        load_data = str_to_object(args.dataloader)
        train_iterator, test_iterator = load_data(
            args.batch_size, args.model_kwargs
            )
    elif args.data_type == "synthetic":
        if "mask_percentage" in args.model_kwargs:
            mask_bool = True
        else:
            mask_bool = False
        load_data = str_to_object(args.dataloader)
        if "projection_dim" in args.model_kwargs:
            X_train, C_train, Cond_indices_train, proj_matrix = load_data(
                args.num_batches,
                args.batch_size,
                args.model_kwargs,
                corr=False,
                train=True,
                mask=mask_bool,
            ).get_all_items()
            path_csv = path_save_dir / Path("projection_options.pt")
            print(proj_matrix)
            with path_csv.open("wb") as fo:
                torch.save(proj_matrix, fo)
            LOGGER.info(f"Saved: {path_csv}")
            X_test, C_test, Cond_indices_test = load_data(
                args.num_batches,
                args.batch_size,
                args.model_kwargs,
                corr=False,
                train=False,
                P=proj_matrix,
                mask=mask_bool,
            ).get_all_items()
        else:
            X_train, C_train, Cond_indices_train, _ = load_data(
                args.num_batches,
                args.batch_size,
                args.model_kwargs,
                corr=False,
                train=True,
                mask=mask_bool,
            ).get_all_items()
            X_test, C_test, Cond_indices_test = load_data(
                args.num_batches,
                args.batch_size,
                args.model_kwargs,
                corr=False,
                train=False,
                mask=mask_bool,
            ).get_all_items()

    model = get_model(args.model_fn, args.model_kwargs).to(device)
    # print('model', model)
    opt = optim.Adam(model.parameters(), lr=args.lr)
    # print('optimizer', opt)
    loss_fn = str_to_object(args.loss_fn)

    if args.data_type == "mnist":
        run = str_to_object(
            "CVAE_testbed.run_models.run_test_train.run_test_train"
            )

        stats = run(
            model,
            opt,
            loss_fn,
            device,
            args.batch_size,
            train_iterator,
            test_iterator,
            args.n_epochs,
            args.model_kwargs,
        )
    elif args.data_type == "synthetic":
        # run = str_to_object('run_models.run_synthetic.run_synthetic')
        run = str_to_object(
            "CVAE_testbed.run_models.run_synthetic.run_synthetic"
            )
        stats, stats_per_dim = run(
            args,
            X_train,
            C_train,
            Cond_indices_train,
            X_test,
            C_test,
            Cond_indices_test,
            args.n_epochs,
            args.loss_fn,
            model,
            opt,
            args.batch_size,
            args.gpu_id,
            args.model_kwargs,
        )
        make_plot_encoding(args, model, stats)

    # print(model, opt)
    this_model = ModelLoader(model, path_save_dir)
    this_model.save_model()
    # save_model(model, path_save_dir)
    path_csv = path_save_dir / Path("costs.csv")
    stats.to_csv(path_csv)
    LOGGER.info(f"Saved: {path_csv}")

    path_csv = path_save_dir / Path("costs_per_dimension.csv")
    stats_per_dim.to_csv(path_csv)
    LOGGER.info(f"Saved: {path_csv}")

    make_plot(stats, stats_per_dim, path_save_dir, args)
    LOGGER.info(f"Elapsed time: {time.time() - tic:.2f}")
    print("saved:", path_save_dir)


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
    fig, (ax1, ax, ax2, ax3) = plt.subplots(1, 4, figsize=(7 * 4, 5))
    fig2 = plt.figure(figsize=(12, 10))
    bax = brokenaxes(xlims=((0, 8), (60, 64)), hspace=0.15)

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

    conds = [i for i in range(this_kwargs)]
    if args.post_plot_kwargs["latent_space_colorbar"] == "yes":
        color = this_dataloader.get_color()
    else:
        color = None

    for i in range(len(conds) + 1):
        # print(conds, i)
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
                label=str(this_kwargs - len(conds)),
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
                label=str(this_kwargs - len(conds)),
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

    for i in range(this_kwargs + 1):
        tmp = kl_per_lt.loc[kl_per_lt["num_conds"] == i]
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
            label=str(i)
            )
        # ax2.plot(np.log2(x + 1) , y, label = str(i))
        bax.plot(x, y, label=str(i))
        sns.scatterplot(
            ax=ax3,
            data=tmp,
            x="rcl",
            y="kl_divergence",
            label=str(i)
            )

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


def make_plot(
        df: pd.DataFrame,
        df2: pd.DataFrame,
        path_save_dir: Path,
        args: argparse.Namespace
        ) -> None:
    """Generates and saves training loss plot.
    Parameters
    ----------
    df
        DataFrame with costs for each echo or iteration.
    path_save_dir
        Plot save directory.
    Returns
    -------
    None
    """
    fig, ax = plt.subplots(figsize=(6.5, 5))
    if "test_kld_per_dim" in df2.columns:
        sns.lineplot(
            ax=ax,
            data=df2,
            x="dimension",
            y="test_kld_per_dim",
            hue="num_conds",
            legend="full",
        )
        # for i in range(len(set(df['num_conds']))):
        #     tmp = df.loc[df['num_conds'] == i]
        #     print('minimum RCL', np.sort(tmp['test_rcl'])[0])
        #     print('minimum KLD', np.sort(tmp['test_klds'])[0])
        ax.set_xlabel("Latent dimension")
        ax.set_ylabel("Sum KLD per dimension")
    path_save_fig = path_save_dir / Path("KLD_vs_dimension.png")
    fig.savefig(path_save_fig, bbox_inches="tight")

    if "fid_any_color_any_digit" in df.columns:
        fig, ax = plt.subplots()
        ax.plot(
            df.index,
            df["fid_any_color_any_digit"],
            marker=".",
            label="fid any color and digit",
        )
        ax.plot(
            df.index,
            df["fid_color_red_any_digit"],
            linestyle="--",
            marker="<",
            label="fid color red any digit",
        )
        ax.plot(
            df.index,
            df["fid_any_color_digit_5"],
            linestyle="-.",
            marker="o",
            label="fid any color digit 5",
        )
        ax.plot(
            df.index,
            df["fid_color_red_digit_5"],
            linestyle=":",
            marker="v",
            label="color red digit 5",
        )
        ax.set_xlabel("Epoch")
        ax.set_ylabel("FID score")
        ax.legend()
        path_save_fig = path_save_dir / Path("fid_scores.png")
        fig.savefig(path_save_fig, bbox_inches="tight")
        LOGGER.info(f"Saved: {path_save_fig}")

    if "generated_image_any_color_any_digit" in df.columns:
        df = df.loc[df["epoch"] == args.n_epochs - 1]
        fig, ax = plt.subplots(figsize=(10, 10))

        ax.imshow(
            np.transpose(
                df["generated_image_any_color_any_digit"].item().cpu(),
                (1, 2, 0)
            )
        )
        path_save_fig = path_save_dir / Path("generated_image_any_color_any_digit.png")
        fig.savefig(path_save_fig, bbox_inches="tight")
        LOGGER.info(f"Saved: {path_save_fig}")

    if "real_image_any_color_any_digit" in df.columns:
        fig, ax = plt.subplots(figsize=(10, 10))
        ax.imshow(
                np.transpose(df["real_image_any_color_any_digit"].item().cpu(), (1, 2, 0))
                )
        path_save_fig = path_save_dir / Path("real_image_any_color_any_digit.png")
        fig.savefig(path_save_fig, bbox_inches="tight")
        LOGGER.info(f"Saved: {path_save_fig}")

    if "generated_image_color_red_any_digit" in df.columns:
        fig, ax = plt.subplots(figsize=(10, 10))
        ax.imshow(
            np.transpose(
                df["generated_image_color_red_any_digit"].item().cpu(),
                (1, 2, 0)
            )
        )
        path_save_fig = path_save_dir / Path("generated_image_color_red_any_digit.png")
        fig.savefig(path_save_fig, bbox_inches="tight")
        LOGGER.info(f"Saved: {path_save_fig}")

    if "real_image_color_red_any_digit" in df.columns:
        fig, ax = plt.subplots(figsize=(10, 10))
        ax.imshow(
            np.transpose(df["real_image_color_red_any_digit"].item().cpu(), (1, 2, 0))
        )
        path_save_fig = path_save_dir / Path("real_image_color_red_any_digit.png")
        fig.savefig(path_save_fig, bbox_inches="tight")
        LOGGER.info(f"Saved: {path_save_fig}")

    if "generated_image_any_color_digit_5" in df.columns:
        fig, ax = plt.subplots(figsize=(10, 10))
        ax.imshow(
            np.transpose(
                df["generated_image_any_color_digit_5"].item().cpu(), (1, 2, 0)
            )
        )
        path_save_fig = path_save_dir / Path("generated_image_any_color_digit_5.png")
        fig.savefig(path_save_fig, bbox_inches="tight")
        LOGGER.info(f"Saved: {path_save_fig}")

    if "real_image_any_color_digit_5" in df.columns:
        fig, ax = plt.subplots(figsize=(10, 10))
        ax.imshow(
            np.transpose(df["real_image_any_color_digit_5"].item().cpu(), (1, 2, 0))
        )
        path_save_fig = path_save_dir / Path("real_image_any_color_digit_5.png")
        fig.savefig(path_save_fig, bbox_inches="tight")
        LOGGER.info(f"Saved: {path_save_fig}")

    if "generated_image_color_red_digit_5" in df.columns:
        fig, ax = plt.subplots(figsize=(10, 10))
        ax.imshow(
            np.transpose(
                df["generated_image_color_red_digit_5"].item().cpu(), (1, 2, 0)
            )
        )
        path_save_fig = path_save_dir / Path("generated_image_color_red_digit_5.png")
        fig.savefig(path_save_fig, bbox_inches="tight")
        LOGGER.info(f"Saved: {path_save_fig}")

    if "real_image_color_red_digit_5" in df.columns:
        fig, ax = plt.subplots(figsize=(10, 10))
        ax.imshow(
            np.transpose(df["real_image_color_red_digit_5"].item().cpu(), (1, 2, 0))
        )
        path_save_fig = path_save_dir / Path("real_image_color_red_digit_5.png")
        fig.savefig(path_save_fig, bbox_inches="tight")
        LOGGER.info(f"Saved: {path_save_fig}")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    train_model()