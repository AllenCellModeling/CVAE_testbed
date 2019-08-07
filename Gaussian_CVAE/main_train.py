import argparse
import logging
import json
import os
import importlib
import time
import pathlib
from pathlib import Path
import datetime
import matplotlib.pyplot as plt
import mpl_toolkits
from mpl_toolkits.axes_grid1 import make_axes_locatable

from torch import nn
from torch import optim
from torch.utils.data import DataLoader
import torch
from torchvision import transforms
from typing import Optional, Dict
from run_models.run_test_train import run_test_train
import pandas as pd
import numpy as np

import seaborn as sns

LOGGER = logging.getLogger(__name__)

def get_args():

    def load_synthetic(x):
        with open(x) as fp:
            a = json.load(fp)
        return a

    parser = argparse.ArgumentParser()
    parser.add_argument('--batch_size', type=int, default=50, help='Mini-batch size')
    parser.add_argument('--dataloader', default='datasets.dataloader.load_mnist_data', help='Data set to load')
    parser.add_argument('--gpu_id', type=int, help='GPU ID')
    parser.add_argument('--loss_fn', default='losses.ELBO.calculate_loss', help='loss_function')
    parser.add_argument('--lr', type=float, default=0.001, help='Learning rate')
    parser.add_argument('--model_fn', default='models.CVAE_first.CVAE', help=', Model name')
    parser.add_argument('--model_kwargs', type=lambda x: load_synthetic(x), default={'x_dim': 2, 'h_dim1':100, 'h_dim2':100, 'z_dim':2, 'c_dim': 2}, help='Model kwargs')
    parser.add_argument('--n_epochs', type=int, default=10, help='Number of training epochs')
    parser.add_argument('--data_type', default='mnist', help='mnist or synthetic data')
    parser.add_argument('--path_save_dir', help='Model save directory')
    args = parser.parse_args()

    # args.path_save_dir = (
    #     args.path_save_dir or str(
    #         Path('outputs', datetime.datetime.today().strftime('%Y:%m:%d:%H:%M:%S') )
    #     ))
    args.path_save_dir = (
        args.path_save_dir + str(datetime.datetime.today().strftime('%Y:%m:%d:%H:%M:%S') ))
    return args

def save_args(args: argparse.Namespace, path_json: Optional[Path] = None) -> None:
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
        path_json = Path(args.path_save_dir) / Path('arguments.json')
    with path_json.open('w') as fo:
        json.dump(vars(args), fo, indent=4)
    LOGGER.info(f'Saved: {path_json}')

def save_model(model: nn.Module, path_save_dir: Path) -> None:
    """Saves model weights and metadata in specified directory."""

    path_save_dir.mkdir(parents=True, exist_ok=True)
    path_weights = path_save_dir / Path('weights.pt')
    device = next(model.parameters()).device  # Get device from first param
    model.to(torch.device('cpu'))
    torch.save(model.state_dict(), path_weights)
    LOGGER.info(f'Saved model weights: {path_weights}')
    model.to(device)

def get_model(model_fn, model_kwargs: Optional[Dict] = None) -> nn.Module:
    model_fn = str_to_object(model_fn)
    model_kwargs = model_kwargs or {}
    # print(model_kwargs)
    # print(model_fn(**model_kwargs))
    return model_fn(**model_kwargs)

def str_to_object(str_o: str) -> object:
    """Get object from string.
    Parameters
    ----------
    str_o
        Fully qualified object name.
    Returns
    -------
    object
        Some Python object.
    """
    parts = str_o.split('.')
    if len(parts) == 1:
        return inspect.currentframe().f_back.f_globals[str_o]
    module = importlib.import_module('.'.join(parts[:-1]))
    return getattr(module, parts[-1])

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
        raise ValueError(f'Save directory already exists! ({path_save_dir})')  
    path_save_dir.mkdir(parents=True)

    logging.getLogger().addHandler(logging.FileHandler(path_save_dir / Path('run.log'), mode='w'))
    save_args(args, path_save_dir / Path('training_options.json'))

    device = (
        torch.device('cuda', args.gpu_id) if torch.cuda.is_available()
        else torch.device('cpu')
    )
    LOGGER.info(f'Using device: {device}')

    if args.data_type == 'mnist':
        load_data = str_to_object(args.dataloader)
        train_iterator, test_iterator = load_data(args.batch_size, args.model_kwargs)
    elif args.data_type == 'synthetic':
        load_data = str_to_object(args.dataloader)
        all_input_train, all_mask_train = load_data(100, args.batch_size, [], args.model_kwargs)
        all_input_test, all_mask_test = load_data(100, args.batch_size, [], args.model_kwargs)
    # print(args.model_fn)

    model = get_model(args.model_fn, args.model_kwargs).to(device)
    # print('model', model)
    opt = optim.Adam(model.parameters(), lr=args.lr)
    # print('optimizer', opt)
    loss_fn = str_to_object(args.loss_fn)

    if args.data_type == 'mnist':
        run = str_to_object('run_models.run_test_train.run_test_train')
        print(model, opt, loss_fn, device, args.batch_size, args.n_epochs, args.model_kwargs)
        stats = run(model, opt, loss_fn, device, args.batch_size, train_iterator, test_iterator, 
                                args.n_epochs, args.model_kwargs)
    elif args.data_type == 'synthetic':
        run = str_to_object('run_models.run_synthetic.run_synthetic')
        stats = run(args, all_input_train, all_mask_train, all_input_test, all_mask_test, 
                    args.n_epochs, args.loss_fn, model, opt, args.batch_size, args.gpu_id, args.model_kwargs)
        make_plot_encoding(args, model)

    # print(model, opt)
    save_model(model, path_save_dir)
    path_csv = path_save_dir / Path('costs.csv')
    stats.to_csv(path_csv)
    LOGGER.info(f'Saved: {path_csv}')
    make_plot(stats, path_save_dir, args)
    LOGGER.info(f'Elapsed time: {time.time() - tic:.2f}')

def make_plot_encoding(args: argparse.Namespace, model) -> None:

    path_save_dir = Path(args.path_save_dir)
    vis_enc = str_to_object('metrics.visualize_encoder.visualize_encoder_synthetic')
    kl_per_dim = str_to_object('metrics.visualize_encoder.get_sorted_klds')
    conds = [i for i in range(args.model_kwargs['x_dim'])]
    fig, ax = plt.subplots(figsize=(6.5,5))
    fig2, ax2 = plt.subplots(figsize=(6.5,5))
    fig3, ax3 = plt.subplots(figsize=(6.5,5))
    
    for j in range(args.model_kwargs['x_dim'] + 1):
        z_means_x, z_means_y, kl_per_lt, fifth_kl_per_lt, z_var_x, z_var_y = vis_enc(args, model, conds, args.batch_size, args.gpu_id, args.model_kwargs)
        kld_avg_dim = kl_per_dim(kl_per_lt)
        kld_avg_dim_5 = kl_per_dim(fifth_kl_per_lt)
        # print(len(z_means_x), len(z_means_y))
        ax.scatter(z_means_x, z_means_y, marker='.', s = 20, label = str(len(conds)))

        fig4, ax4 = plt.subplots(figsize=(6.5,5))
        scatters4 = ax4.scatter(z_means_x, z_means_y, marker='.', s=20, c=z_var_x, cmap = 'inferno')
        colorbar(scatters4)
        path_save_fig = path_save_dir / Path('latent_space_colormap_variance_1_conds' + str(len(conds)) + '.png')
        fig4.savefig(path_save_fig, bbox_inches='tight')

        fig5, ax5 = plt.subplots(figsize=(6.5,5))
        scatters5 = ax5.scatter(z_means_x, z_means_y, marker='.', s=20, c=z_var_y, cmap = 'inferno')
        colorbar(scatters5)
        path_save_fig = path_save_dir / Path('latent_space_colormap_variance_2_conds' + str(len(conds)) + '.png')  
        fig5.savefig(path_save_fig, bbox_inches='tight')
        
        if j == 0:
            sort_x = np.sort(z_means_x)
            sort_y = np.sort(z_means_y)
            ax.set_xlim([sort_x[0], sort_x[-1]])
            ax.set_ylim([sort_y[0], sort_y[-1]])
        ax2.plot(kld_avg_dim, label = str(len(conds)))
        ax2.set_xlabel('Latent dimension')
        ax2.set_ylabel('Average KL divergence')

        ax3.plot(kld_avg_dim_5, label = str(len(conds)))
        ax3.set_xlabel('Latent dimension')
        ax3.set_ylabel('Average KL divergence')
        # print(conds)
        try:
            conds.pop()
        except: 
            pass
    ax.legend()
    ax2.legend()
    path_save_fig = path_save_dir / Path('latent_space.png')
    fig.savefig(path_save_fig, bbox_inches='tight')
    LOGGER.info(f'Saved: {path_save_fig}')
    path_save_fig = path_save_dir / Path('KLD_per_dim_last_epoch.png')
    fig2.savefig(path_save_fig, bbox_inches='tight')
    LOGGER.info(f'Saved: {path_save_fig}')
    path_save_fig = path_save_dir / Path('KLD_per_dim_fifth_epoch.png')
    fig3.savefig(path_save_fig, bbox_inches='tight')
    LOGGER.info(f'Saved: {path_save_fig}')

def make_plot(df: pd.DataFrame, path_save_dir: Path, args: argparse.Namespace) -> None:
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
    fig, ax = plt.subplots(figsize=(6.5,5))
    fig2, ax2 = plt.subplots(figsize=(6.5,5))
    if 'train_losses' in df.columns:
        sns.lineplot(ax =ax, data = df, x = 'epoch', y = 'train_losses', hue = 'num_conds')
        ax.set_xlabel('Epoch')
        ax.set_ylabel('Train loss')
        max_ylim = np.sort(df['test_losses'].values)[-1] + 0.4
        # ax.set_ylim([0, max_ylim])
    if 'test_losses' in df.columns:
        sns.lineplot(ax =ax2, data = df, x = 'epoch', y = 'test_losses', hue = 'num_conds')
        ax2.set_xlabel('Epoch')
        ax2.set_ylabel('Test loss')
        max_ylim = np.sort(df['test_losses'].values)[-1] + 0.4
        # ax2.set_ylim([0, max_ylim])
    path_save_fig = path_save_dir / Path('train_costs.png')
    fig.savefig(path_save_fig, bbox_inches='tight')

    path_save_fig = path_save_dir / Path('test_costs.png')
    fig2.savefig(path_save_fig, bbox_inches='tight')
    LOGGER.info(f'Saved: {path_save_fig}')

    fig, ax = plt.subplots(figsize=(6.5,5))
    if 'test_rcl' in df.columns:
        sns.lineplot(ax = ax, data = df, x = 'test_rcl', y= 'test_klds', hue = 'num_conds')
        print('test rcl in columns')
        for i in range(len(set(df['num_conds']))):
            tmp = df.loc[df['num_conds'] == i]
            print('minimum RCL', np.sort(tmp['test_rcl'])[0])
            print('minimum KLD', np.sort(tmp['test_klds'])[0])
        ax2.set_xlabel('MSE')
        ax2.set_ylabel('KLD')
        max_ylim = np.sort(df['test_klds'].values)[-1] + 0.4
    path_save_fig = path_save_dir / Path('KLD_vs_MSE.png')
    fig.savefig(path_save_fig, bbox_inches='tight')

    if 'fid_any_color_any_digit' in df.columns:
        fig, ax = plt.subplots()
        ax.plot(df.index, df['fid_any_color_any_digit'], marker='.', label='fid any color and digit')
        ax.plot(df.index, df['fid_color_red_any_digit'], linestyle='--', marker='<', label='fid color red any digit')
        ax.plot(df.index, df['fid_any_color_digit_5'], linestyle='-.', marker='o', label='fid any color digit 5')
        ax.plot(df.index, df['fid_color_red_digit_5'], linestyle=':', marker='v', label='color red digit 5')
        ax.set_xlabel('Epoch')
        ax.set_ylabel('FID score')
        ax.legend()
        path_save_fig = path_save_dir / Path('fid_scores.png')
        fig.savefig(path_save_fig, bbox_inches='tight')
        LOGGER.info(f'Saved: {path_save_fig}')

    if 'generated_image_any_color_any_digit' in df.columns:
        df = df.loc[df['epoch'] == args.n_epochs-1]
        fig, ax = plt.subplots(figsize = (10, 10))

        ax.imshow(np.transpose(df['generated_image_any_color_any_digit'].item().cpu(), (1,2, 0)))      
        path_save_fig = path_save_dir / Path('generated_image_any_color_any_digit.png')
        fig.savefig(path_save_fig, bbox_inches='tight')
        LOGGER.info(f'Saved: {path_save_fig}')

    if 'real_image_any_color_any_digit' in df.columns:
        fig, ax = plt.subplots(figsize = (10, 10))
        ax.imshow(np.transpose(df['real_image_any_color_any_digit'].item().cpu(), (1,2, 0)))      
        path_save_fig = path_save_dir / Path('real_image_any_color_any_digit.png')
        fig.savefig(path_save_fig, bbox_inches='tight')
        LOGGER.info(f'Saved: {path_save_fig}')
    
    if 'generated_image_color_red_any_digit' in df.columns:
        fig, ax = plt.subplots(figsize = (10, 10))
        ax.imshow(np.transpose(df['generated_image_color_red_any_digit'].item().cpu(), (1,2, 0)))      
        path_save_fig = path_save_dir / Path('generated_image_color_red_any_digit.png')
        fig.savefig(path_save_fig, bbox_inches='tight')
        LOGGER.info(f'Saved: {path_save_fig}')

    if 'real_image_color_red_any_digit' in df.columns:
        fig, ax = plt.subplots(figsize = (10, 10))
        ax.imshow(np.transpose(df['real_image_color_red_any_digit'].item().cpu(), (1,2, 0)))      
        path_save_fig = path_save_dir / Path('real_image_color_red_any_digit.png')
        fig.savefig(path_save_fig, bbox_inches='tight')
        LOGGER.info(f'Saved: {path_save_fig}')

    if 'generated_image_any_color_digit_5' in df.columns:
        fig, ax = plt.subplots(figsize = (10, 10))
        ax.imshow(np.transpose(df['generated_image_any_color_digit_5'].item().cpu(), (1,2, 0)))      
        path_save_fig = path_save_dir / Path('generated_image_any_color_digit_5.png')
        fig.savefig(path_save_fig, bbox_inches='tight')
        LOGGER.info(f'Saved: {path_save_fig}')

    if 'real_image_any_color_digit_5' in df.columns:
        fig, ax = plt.subplots(figsize = (10, 10))
        ax.imshow(np.transpose(df['real_image_any_color_digit_5'].item().cpu(), (1,2, 0)))      
        path_save_fig = path_save_dir / Path('real_image_any_color_digit_5.png')
        fig.savefig(path_save_fig, bbox_inches='tight')
        LOGGER.info(f'Saved: {path_save_fig}')

    if 'generated_image_color_red_digit_5' in df.columns:
        fig, ax = plt.subplots(figsize = (10, 10))
        ax.imshow(np.transpose(df['generated_image_color_red_digit_5'].item().cpu(), (1,2, 0)))      
        path_save_fig = path_save_dir / Path('generated_image_color_red_digit_5.png')
        fig.savefig(path_save_fig, bbox_inches='tight')
        LOGGER.info(f'Saved: {path_save_fig}')

    if 'real_image_color_red_digit_5' in df.columns:
        fig, ax = plt.subplots(figsize = (10, 10))
        ax.imshow(np.transpose(df['real_image_color_red_digit_5'].item().cpu(), (1,2, 0)))      
        path_save_fig = path_save_dir / Path('real_image_color_red_digit_5.png')
        fig.savefig(path_save_fig, bbox_inches='tight')
        LOGGER.info(f'Saved: {path_save_fig}')

if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)
    train_model()

