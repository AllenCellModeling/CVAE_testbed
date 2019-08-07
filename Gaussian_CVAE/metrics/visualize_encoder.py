import torch
import numpy as np
from losses.ELBO import synthetic_loss
import pandas as pd
from datasets.dataloader import make_synthetic_data
from main_train import str_to_object

def visualize_encoder_synthetic(args, model, conds, batch_size, gpu_id, model_kwargs):

    z_means_x, z_means_y = [], []
    z_var_x, z_var_y = [], []
    num_batches = args.batch_size
    model.cuda(gpu_id)

    # kl_vs_lt_all_epochs = {'epoch': [], 'kl_per_lt': []}
    with torch.no_grad():
        for i in range(num_batches):
            make_data = str_to_object(args.dataloader)
            # print(args.dataloader)
            # print('inside visualization encoder')
            # print(conds)
            c, d = make_data(1, batch_size, conds, model_kwargs)  
            c = c[0, :]
            d = d[0, :]
            if args.dataloader != 'datasets.dataloader.make_synthetic_data_3': 
                # print(args.dataloader)
                if len(conds) > 0:
                    tmp1, tmp2 = torch.split(d, 2, dim=1)
                    for kk in conds:
                        tmp1[:, kk], tmp2[:, kk] = 0, 0
                    d = torch.cat((tmp1, tmp2), 1)
            else:
                for kk in conds:
                    d[:, kk] =0
            # print(c.size(), d.size()) 
            recon_batch, z_means, log_var = model(c.cuda(gpu_id), d.cuda(gpu_id))
            all_kl, all_lt = [], []
            kl_per_lt = {'latent_dim': [], 'kl_divergence': []}
            loss_fn = str_to_object(args.loss_fn)
            loss_whole_batch, batch_rcl, batch_kl = loss_fn(c.cuda(gpu_id), recon_batch.cuda(gpu_id), z_means, log_var)
            # print('Batch loss')
            # print(loss_whole_batch.item(), batch_rcl.item(), batch_kl.item())
            # print('conds')
            # print(conds)
            for ii in range(z_means.size()[-1]):
                loss_fn = str_to_object(args.loss_fn)
                # print(args.loss_fn)
                total_loss, rcl, kl_per_lt_temp = loss_fn(c.cuda(gpu_id), recon_batch.cuda(gpu_id), z_means[:, ii], log_var[:, ii])
                
                # print(total_loss.item(), rcl.item(), kl_per_lt_temp.item())
                all_kl = np.append(all_kl, kl_per_lt_temp.item())
                all_lt.append(ii)
                kl_per_lt['latent_dim'].append(ii)
                kl_per_lt['kl_divergence'].append(kl_per_lt_temp.item())
            all_kl, all_lt = list(zip(*sorted(zip(all_kl, all_lt))))
            all_kl = list(all_kl)
            all_lt = list(all_lt)

            if i == 5:
                fifth_kl_per_lt = kl_per_lt
            # kl_vs_lt_all_epochs['epoch'].append(i)
            # kl_vs_lt_all_epochs['kl_per_lt'].append(kl_per_lt)

            z_means_x = np.append(z_means_x, z_means[:, all_lt[-1]].data.cpu().numpy())
            z_means_y = np.append(z_means_y, z_means[:, all_lt[-2]].data.cpu().numpy())
            z_var_x = np.append(z_var_x, log_var[:, all_lt[-1]].data.cpu().numpy())
            z_var_y = np.append(z_var_y, log_var[:, all_lt[-2]].data.cpu().numpy())
    # kl_vs_lt_all_epochs_dataframe = pd.DataFrame(kl_vs_lt_all_epochs)
    return z_means_x, z_means_y, kl_per_lt, fifth_kl_per_lt, z_var_x, z_var_y

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