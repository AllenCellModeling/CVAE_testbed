import argparse
import logging
import json
import time
from pathlib import Path
import datetime
from mpl_toolkits.axes_grid1 import make_axes_locatable

import torch
from torch import nn, optim

from typing import Optional, Dict
from CVAE_testbed.models.model_loader import ModelLoader
from CVAE_testbed.utils import str_to_object

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
        type=float,
        default=1,
        help="Beta parameter in beta*(RCL + KLD) loss"
    )
    parser.add_argument(
        "--json_quilt_path",
        default='/home/ritvik.vasan/test/',
        help="Path to json files containing quilt features"
    )
    parser.add_argument(
        "--binary_real_one_hot_parameters",
        type=lambda x: load_synthetic(x),
        default={
            "binary_range": [0, 1],
            "binary_loss": 'BCE',
            "real_range": [1, 103],
            "real loss": 'MSE',
            "one_hot_range": [103, 159],
            "one_hot_loss": 'NLL'
        },
        help="Losses and ranges for binary real and one_hot data",
    )
    parser.add_argument(
        "--config_path",
        default='/home/ritvik.vasan/config.json',
        help="Path to config file for Jackson's feature database"
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
            "latent_space_colorbar": "no",
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

    if args.path_save_dir is not None:
        args.path_save_dir = args.path_save_dir
    else: 
        args.path_save_dir = './outputs/' + str(
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
                if key != "sklearn_data" and key != 'projection_dim' and key != 'mask_percentage'
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

    feature_names = None
    proj_matrix = None

    if args.data_type == "mnist":
        load_data = str_to_object(args.dataloader)
        train_iterator, test_iterator = load_data(
            args.batch_size, args.model_kwargs
            )
    elif args.data_type == "aics_features":
        load_data = str_to_object(args.dataloader)
        test_instance = load_data(
            args.num_batches,
            args.batch_size,
            args.model_kwargs,
            corr=False,
            train=True,
            mask=False,
        )
        X_train, C_train, Cond_indices_train = test_instance.get_train_data()
        X_test, C_test, Cond_indices_test = test_instance.get_test_data()
        feature_names = test_instance.get_feature_names()
        this_dataloader_color = test_instance.get_color()
        # print('CVAE train')
        # print(X_train.size(), X_test.size())
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
            test_instance = load_data(
                                        args.num_batches,
                                        args.batch_size,
                                        args.model_kwargs,
                                        corr=False,
                                        train=False,
                                        P=proj_matrix,
                                        mask=mask_bool,
                                     )
            X_test, C_test, Cond_indices_test = test_instance.get_all_items()
            this_dataloader_color = test_instance.get_color()
        else:
            X_train, C_train, Cond_indices_train, _ = load_data(
                args.num_batches,
                args.batch_size,
                args.model_kwargs,
                corr=False,
                train=True,
                mask=mask_bool,
            ).get_all_items()
            test_instance = load_data(
                                        args.num_batches,
                                        args.batch_size,
                                        args.model_kwargs,
                                        corr=False,
                                        train=False,
                                        mask=mask_bool,
                                      )
            X_test, C_test, Cond_indices_test = test_instance.get_all_items()
            this_dataloader_color = test_instance.get_color()

    model = get_model(args.model_fn, args.model_kwargs).to(device)
    opt = optim.Adam(model.parameters(), lr=args.lr)
    loss_fn = str_to_object(args.loss_fn)

    make_plot_encoding_greedy = str_to_object(
            "CVAE_testbed.utils.greedy_encoding_plots.make_plot_encoding_greedy"
            )
    make_plot_encoding = str_to_object(
            "CVAE_testbed.utils.encoding_plots.make_plot_encoding"
            )
    make_plot = str_to_object(
            "CVAE_testbed.utils.loss_and_image_plots.make_plot"
            )
    pca = str_to_object(
            "CVAE_testbed.utils.pca.get_PCA_features"
            )

    make_fid_plot = str_to_object(
            "CVAE_testbed.utils.FID_score.make_plot_FID"
            )

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
    elif args.data_type == "synthetic" or args.data_type == 'aics_features':
        run = str_to_object(
            "CVAE_testbed.run_models.run_synthetic_test.run_synthetic"
            )
        stats, stats_per_dim = run(
            args,
            X_train,
            C_train,
            Cond_indices_train,
            X_test[:-3, :, :],
            C_test[:-3, :, :],
            Cond_indices_test,
            args.n_epochs,
            args.loss_fn,
            model,
            opt,
            args.batch_size,
            args.gpu_id,
            args.model_kwargs,
        )
        if args.data_type == 'aics_features' or args.data_type == 'synthetic':
            print(proj_matrix)

            # First load non shuffled data
            if proj_matrix is not None:
                this_dataloader = load_data(
                                    args.num_batches, args.batch_size, args.model_kwargs, shuffle=False, P = proj_matrix, train=False
                                        )
            elif args.data_type == 'aics_features':
                this_dataloader = load_data(
                                    args.num_batches, args.batch_size, args.model_kwargs, shuffle=False, train=False
                                        )
            else:
                this_dataloader = load_data(
                    args.num_batches, args.batch_size, args.model_kwargs, shuffle=False, train=False
                                        )
            X_non_shuffled, C_non_shuffled, _ = this_dataloader.get_all_items()

            # Now check encoding
            make_plot_encoding(args, model, stats, X_non_shuffled.clone(), C_non_shuffled.clone(), this_dataloader_color, True, proj_matrix)
            pca_dataframe = pca(args, this_dataloader, True)
            make_plot_encoding_greedy(args, model, stats,  X_non_shuffled.clone(), C_non_shuffled.clone(), feature_names, True, proj_matrix)
            try:
                make_fid_plot(args, model,  X_non_shuffled.clone(), C_non_shuffled.clone())
            except:
                pass
        else:
            make_plot_encoding(args, model, stats, X_test, C_test)

    this_model = ModelLoader(model, path_save_dir)
    this_model.save_model()
    path_csv = path_save_dir / Path("costs.csv")
    stats.to_csv(path_csv)
    LOGGER.info(f"Saved: {path_csv}")

    path_csv = path_save_dir / Path("costs_per_dimension.csv")
    stats_per_dim.to_csv(path_csv)
    LOGGER.info(f"Saved: {path_csv}")

    make_plot(stats, stats_per_dim, path_save_dir, args)
    LOGGER.info(f"Elapsed time: {time.time() - tic:.2f}")
    print("saved:", path_save_dir)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    train_model()
