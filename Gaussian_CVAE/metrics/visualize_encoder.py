import torch
import numpy as np
from Gaussian_CVAE.losses.ELBO import synthetic_loss
import pandas as pd
from Gaussian_CVAE.main_train import str_to_object

def visualize_encoder_synthetic(args, model, conds, c, d, kl_per_lt=None):

    z_means_x, z_means_y = [], []
    z_var_x, z_var_y = [], []
    model.cuda(args.gpu_id)

    with torch.no_grad():
        if kl_per_lt is None:
            kl_per_lt = {'latent_dim': [], 'kl_divergence': [], 'num_conds': []}
        z_means_scatterplot_dict = {'z_means_x': [], 'z_means_y': [], 'z_var_x': [], 'z_var_y': [], 'num_conds': []}
        all_kl, all_lt = [], []

        tmp1, tmp2 = torch.split(d, int(d.size()[-1]/2), dim=1)
        for kk in conds:
            tmp1[:, kk], tmp2[:, kk] = 0, 0
        cond_d = torch.cat((tmp1, tmp2), 1)

        recon_batch, z_means, log_var = model(c.cuda(args.gpu_id), cond_d.cuda(args.gpu_id))
        # loss_fn = str_to_object(args.loss_fn)
        # loss_whole_batch, batch_rcl, batch_kld, rcl_per_element, kld_per_element = loss_fn(c.cuda(args.gpu_id), recon_batch.cuda(args.gpu_id), z_means, log_var)
        
        
        for ii in range(z_means.size()[-1]):
            loss_fn = str_to_object(args.loss_fn)
            total_loss, rcl, kl_per_lt_temp, _, _ = loss_fn(c.cuda(args.gpu_id), recon_batch.cuda(args.gpu_id), z_means[:, ii], log_var[:, ii])

            # print(total_loss.item(), rcl.item(), kl_per_lt_temp.item())
            all_kl = np.append(all_kl, kl_per_lt_temp.item())
            all_lt.append(ii)
            kl_per_lt['num_conds'].append(c.size()[-1] - len(conds))
            kl_per_lt['latent_dim'].append(ii)
            kl_per_lt['kl_divergence'].append(kl_per_lt_temp.item())
        all_kl, all_lt = list(zip(*sorted(zip(all_kl, all_lt))))
        all_kl = list(all_kl)
        all_lt = list(all_lt)

            # this_cond_rcl_per_dimension = torch.sort(torch.sum(rcl_per_element[this_cond_positions], dim = 0))[0]
            #     this_cond_kld_per_dimension = torch.sort(torch.sum(kld_per_element[this_cond_positions], dim = 0))[0]
            #     max_kld_index = torch.sort(torch.sum(kld_per_element[this_cond_positions], dim = 0))[1][-1].item()
            #     second_max_kld_index = torch.sort(torch.sum(kld_per_element[this_cond_positions], dim = 0))[1][-2].item()
            #     for k in range(z_means.size()[0]):
            #         z_means_scatterplot_dict['num_conds'].append(int(args.model_kwargs['x_dim'] - ii.item()))
            #         z_means_scatterplot_dict['z_means_x'].append(z_means[k, max_kld_index].item())
            #         z_means_scatterplot_dict['z_means_y'].append(z_means[k, second_max_kld_index].item())
            #         z_means_scatterplot_dict['z_var_x'].append(log_var[k, max_kld_index].item())
            #         z_means_scatterplot_dict['z_var_y'].append(log_var[k, second_max_kld_index].item())           
            #     for index, mm in enumerate(range(args.model_kwargs['dec_layers'][0])):
            #         kl_per_lt['latent_dim'].append(index)
            #         kl_per_lt['kl_divergence'].append(this_cond_kld_per_dimension[mm].item())
            #         kl_per_lt['num_conds'].append(int(args.model_kwargs['x_dim'] - ii.item()))
            
        z_means_x = np.append(z_means_x, z_means[:, all_lt[-1]].data.cpu().numpy())
        z_means_y = np.append(z_means_y, z_means[:, all_lt[-2]].data.cpu().numpy())
        z_var_x = np.append(z_var_x, log_var[:, all_lt[-1]].data.cpu().numpy())
        z_var_y = np.append(z_var_y, log_var[:, all_lt[-2]].data.cpu().numpy())
            # kl_vs_lt_all_epochs_dataframe = pd.DataFrame(kl_vs_lt_all_epochs)
    return z_means_x, z_means_y, kl_per_lt, z_var_x, z_var_y
                
    # z_means_scatterplot_dict = pd.DataFrame(z_means_scatterplot_dict)
    # kl_per_lt = pd.DataFrame(kl_per_lt)

    # return z_means_scatterplot_dict, kl_per_lt

def get_sorted_klds(latent_dict):
    kl_data = {'n_dim': [], 'kld_avg_dim': []}

    df = pd.DataFrame(latent_dict)
    df = df.sort_values(by=['kl_divergence'])
    n_dim = np.max(df['latent_dim'])

    kld_avg_dim = np.zeros(n_dim)

    for i in range(n_dim):
        kld_avg_dim[i] = np.mean(df['kl_divergence'][df['latent_dim'] == i])
    kld_avg_dim = np.sort(kld_avg_dim)[::-1]

    return kld_avg_dim