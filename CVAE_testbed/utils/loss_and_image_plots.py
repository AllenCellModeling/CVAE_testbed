import argparse
import logging
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

LOGGER = logging.getLogger(__name__)


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