from Gaussian_CVAE.run_models.train import train
from Gaussian_CVAE.run_models.test import test
import torch
from Gaussian_CVAE.run_models.generative_metric import compute_generative_metric
import pandas as pd
import numpy as np


def run_test_train(model, optimizer, loss_fn, device, BATCH_SIZE, train_iterator, test_iterator, n_epochs, LATENT_DIM):

    dataframe = {'epoch': [], 'train_losses': [], 'train_rcl_losses': [], 'train_kl_losses': [], 'test_losses': [],
             'test_rcl_losses': [], 'test_kl_losses': [], 'fid_any_color_any_digit': [],
             'fid_color_red_any_digit': [], 'fid_any_color_digit_5': [], 
             'fid_color_red_digit_5': [], 'real_image_color_red_digit_5': [], 'generated_image_color_red_digit_5': [], 
             'real_image_any_color_digit_5': [], 'generated_image_any_color_digit_5': [], 
             'real_image_color_red_any_digit': [], 'generated_image_color_red_any_digit': [],
              'real_image_any_color_any_digit': [], 'generated_image_any_color_any_digit': []}

    for e in range(n_epochs):
        train_loss, tr_rcl_loss, tr_kld_loss, tr_kl_per_lt = train(model, optimizer, train_iterator, device, BATCH_SIZE)
        test_loss, test_rcl_loss, test_kld_loss, test_kl_per_lt, blur = test(model, optimizer, test_iterator, device, BATCH_SIZE)
        with torch.no_grad():
            fid1, grid1_1, grid1_2 = compute_generative_metric(test_iterator, model, device, LATENT_DIM['latent_dim'], BATCH_SIZE, 
                                        color_value= None, digit_value=None)
            fid2, grid2_1, grid2_2 = compute_generative_metric(test_iterator, model, device, LATENT_DIM['latent_dim'], BATCH_SIZE, 
                                        color_value= 0, digit_value=None)
            fid3, grid3_1, grid3_2 = compute_generative_metric(test_iterator, model, device, LATENT_DIM['latent_dim'], BATCH_SIZE, 
                                        color_value= None, digit_value=5)
            fid4, grid4_1, grid4_2 = compute_generative_metric(test_iterator, model, device, LATENT_DIM['latent_dim'], BATCH_SIZE, 
                                        color_value= 0, digit_value=5)

        
        train_dataset_size = len(train_iterator)*BATCH_SIZE
        test_dataset_size = len(test_iterator)*BATCH_SIZE
        train_loss /= train_dataset_size
        tr_rcl_loss /= train_dataset_size
        tr_kld_loss /= train_dataset_size
        test_loss /= test_dataset_size
        test_rcl_loss /= test_dataset_size
        test_kld_loss /= test_dataset_size

        print(train_loss, test_loss)

        dataframe['epoch'].append(e)
        dataframe['train_losses'].append(train_loss)
        dataframe['train_rcl_losses'].append(tr_rcl_loss)
        dataframe['train_kl_losses'].append(tr_kld_loss)
        dataframe['test_losses'].append(test_loss)
        dataframe['test_rcl_losses'].append(test_rcl_loss)
        dataframe['test_kl_losses'].append(test_kld_loss)

        dataframe['fid_any_color_any_digit'].append(fid1)
        dataframe['real_image_any_color_any_digit'].append(grid1_1)
        dataframe['generated_image_any_color_any_digit'].append(grid1_2)
    
        dataframe['fid_color_red_any_digit'].append(fid2)
        dataframe['real_image_color_red_any_digit'].append(grid2_1)
        dataframe['generated_image_color_red_any_digit'].append(grid2_2)

        dataframe['fid_any_color_digit_5'].append(fid3)
        dataframe['real_image_any_color_digit_5'].append(grid3_1)
        dataframe['generated_image_any_color_digit_5'].append(grid3_2)

        dataframe['fid_color_red_digit_5'].append(fid4)
        dataframe['real_image_color_red_digit_5'].append(grid4_1)
        dataframe['generated_image_color_red_digit_5'].append(grid4_2)

    stats = pd.DataFrame(dataframe)

    return stats