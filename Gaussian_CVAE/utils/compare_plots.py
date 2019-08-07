import torch 
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
import pathlib
from pathlib import Path
sns.set_context('paper')
sns.set(font_scale = 1) 
palette = sns.color_palette("mako_r", 10)

def compare_plots_best_performing(csv):

    compare_plot_df = {'model': [], 'conds': [], 'rate': [], 'distortion': []}

    fig, ax = plt.subplots(figsize = (8,6))
    for j in range(len(csv['csv_path'])):
        # read the j'th csv that we want to compare
        csv_df = pd.read_csv(csv['csv_path'][j])
        # each csv has a number of conditions
        # Lets loop throught those
        for i in range(len(set(csv_df['num_conds']))):

            tmp = csv_df.loc[csv_df['num_conds'] == i]
            compare_plot_df['model'].append(j)
            compare_plot_df['conds'].append(i)
            compare_plot_df['rate'].append(np.sort(tmp['test_klds'])[0])
            compare_plot_df['distortion'].append(np.sort(tmp['test_rcl'])[0])

        dataframe_for_lineplot = pd.DataFrame(compare_plot_df)
        dataframe_for_lineplot = dataframe_for_lineplot.loc[dataframe_for_lineplot['model'] == j]
        sns.lineplot(ax = ax, data = dataframe_for_lineplot, x = 'distortion', y = 'rate', color = 'black', lw = 0.5)
    compare_plot_df = pd.DataFrame(compare_plot_df)

    sns.scatterplot(ax = ax, data = compare_plot_df, x = 'distortion', y = 'rate', hue = 'conds',
                style = 'model', s=200)

    plt.legend(loc='upper left')
    ax.set_xlabel('Distortion')
    ax.set_ylabel('Rate')
    fig.savefig('./multiple_models_conds.png', bbox_inches='tight')

def plot_single_model_multiple_epoch(csv, compare_plot_df = None, count = None, 
                                    fig=None, ax=None, total=None, save=True):

    if compare_plot_df is None:
        fig, ax = plt.subplots(figsize = (8,6))
        compare_plot_df = {'epoch': [], 'conds': [], 'rate': [], 'distortion': []}
        csv_df = pd.read_csv(csv['csv_path'])
    else:
        csv_df = pd.read_csv(csv['csv_path'][0])

    # each csv has a number of conditions
    # Lets loop throught those
    for k in range(len(set(csv_df['epoch']))):
        # Plot only odd epochs
        if k % 2 != 0:
            for i in range(len(set(csv_df['num_conds']))):

                tmp = csv_df.loc[csv_df['num_conds'] == i]
                if count is not None:
                    compare_plot_df['model'].append(count)

                compare_plot_df['epoch'].append(k)
                compare_plot_df['conds'].append(i)
                compare_plot_df['rate'].append(tmp.loc[tmp['epoch'] == k]['test_klds'].item())
                compare_plot_df['distortion'].append(tmp.loc[tmp['epoch'] == k]['test_rcl'].item())

    compare_plot_df = pd.DataFrame(compare_plot_df)

    if save is True:
        sns.scatterplot(ax = ax, data = compare_plot_df, x = 'distortion', y = 'rate',
                        hue = 'conds', s=200)
    else:
        if count != total:
            sns.scatterplot(ax = ax, data = compare_plot_df, x = 'distortion', y = 'rate',
                            hue = 'conds', style = 'model', s=200, legend=False)
        else:
            sns.scatterplot(ax = ax, data = compare_plot_df, x = 'distortion', y = 'rate', hue = 'conds', style = 'model', 
                            s=200)

    if count is not None:
        compare_plot_df = compare_plot_df.loc[compare_plot_df['model'] == count]
        if count == 1:
            sns.lineplot(ax = ax, data = compare_plot_df, x = 'distortion', y = 'rate', color = 'black',
                        hue = 'epoch',palette="ch:2.5,.25")
        else:
            sns.lineplot(ax = ax, data = compare_plot_df, x = 'distortion', y = 'rate', color = 'black',
                        hue = 'epoch',palette="ch:2.5,.25", legend=False)
    else:
        sns.lineplot(ax = ax, data = compare_plot_df, x = 'distortion', y = 'rate', color = 'black',
                 hue = 'epoch',palette="ch:2.5,.25")


    plt.legend(loc='upper left')
    ax.set_xlabel('Distortion')
    ax.set_ylabel('Rate')
    if save is True:
        fig.savefig('./multiple_conds_epochs.png', bbox_inches='tight')
    return fig, ax

def plot_multiple_model_multiple_epoch(csv):
    compare_plot_df = {'model': [], 'epoch': [], 'conds': [], 'rate': [], 'distortion': []}

    fig, ax = plt.subplots(figsize = (8,6))
    count = 1
    total = len(csv['csv_path'])
    for i, j in zip(csv['save_dir'], csv['csv_path']):
        this_csv = {'save_dir': [], 'csv_path': []}
        this_csv['save_dir'].append(i)
        this_csv['csv_path'].append(j)

        fig, ax = plot_single_model_multiple_epoch(this_csv, compare_plot_df, count, fig, ax, total, save=False)
        count += 1
    fig.savefig('./multiple_models_conds_epochs.png', bbox_inches='tight')



