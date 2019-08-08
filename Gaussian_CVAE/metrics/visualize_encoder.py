import torch
import numpy as np
from losses.ELBO import synthetic_loss
import pandas as pd
from datasets.dataloader import make_synthetic_data
from main_train import str_to_object

def visualize_encoder_synthetic(args, model):

    z_means_x, z_means_y = [], []
    z_var_x, z_var_y = [], []
    model.cuda(args.gpu_id)

    with torch.no_grad():
        kl_per_lt = {'latent_dim': [], 'kl_divergence': [], 'num_conds': []}
        z_means_scatterplot_dict = {'z_means_x': [], 'z_means_y': [], 'z_var_x': [], 'z_var_y': [], 'num_conds': []}

        for i in range(50):
            make_data = str_to_object(args.dataloader)

            c, d, ind = make_data(1, args.batch_size, args.model_kwargs)
            c = c[0, :]
            d = d[0, :]
            ind = ind[0, :]
            # if args.dataloader != 'datasets.dataloader.make_synthetic_data_3': 
            #     # print(args.dataloader)
            #     if len(conds) > 0:
            #         tmp1, tmp2 = torch.split(d, 2, dim=1)
            #         for kk in conds:
            #             tmp1[:, kk], tmp2[:, kk] = 0, 0
            #         d = torch.cat((tmp1, tmp2), 1)
            # else:
            #     for kk in conds:
            #         d[:, kk] =0
            # print(c.size(), d.size()) 
            recon_batch, z_means, log_var = model(c.cuda(args.gpu_id), d.cuda(args.gpu_id))
            loss_fn = str_to_object(args.loss_fn)
            loss_whole_batch, batch_rcl, batch_kld, rcl_per_element, kld_per_element = loss_fn(c.cuda(args.gpu_id), recon_batch.cuda(args.gpu_id), z_means, log_var)
            

            for jj, ii in enumerate(torch.unique(ind)):
                this_cond_positions = ind == i
                this_cond_rcl_per_dimension = torch.sort(torch.sum(rcl_per_element[this_cond_positions], dim = 0))[0]
                this_cond_kld_per_dimension = torch.sort(torch.sum(kld_per_element[this_cond_positions], dim = 0))[0]
                max_kld_index = torch.sort(torch.sum(kld_per_element[this_cond_positions], dim = 0))[1][-1].item()
                second_max_kld_index = torch.sort(torch.sum(kld_per_element[this_cond_positions], dim = 0))[1][-2].item()
                for k in range(z_means.size()[0]):
                    z_means_scatterplot_dict['num_conds'].append(args.model_kwargs['x_dim'] - ii.item())
                    z_means_scatterplot_dict['z_means_x'].append(z_means[k, max_kld_index].item())
                    z_means_scatterplot_dict['z_means_y'].append(z_means[k, second_max_kld_index].item())
                    z_means_scatterplot_dict['z_var_x'].append(log_var[k, max_kld_index].item())
                    z_means_scatterplot_dict['z_var_y'].append(log_var[k, second_max_kld_index].item())           
                for index, mm in enumerate(range(args.model_kwargs['dec_layers'][0])):
                    kl_per_lt['latent_dim'].append(index)
                    kl_per_lt['kl_divergence'].append(this_cond_kld_per_dimension[mm].item())
                    kl_per_lt['num_conds'].append(args.model_kwargs['x_dim'] - ii.item())
                
    z_means_scatterplot_dict = pd.DataFrame(z_means_scatterplot_dict)
    kl_per_lt = pd.DataFrame(kl_per_lt)

    return z_means_scatterplot_dict, kl_per_lt

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