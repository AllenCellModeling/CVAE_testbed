import torch
import numpy as np
from CVAE_testbed.utils import str_to_object
import pandas as pd


def visualize_encoder_synthetic(args, model, conds, c, d, kl_per_lt=None, kl_vs_rcl=None):

    z_means_x, z_means_y = [], []
    z_var_x, z_var_y = [], []
    model.cuda(args.gpu_id)

    with torch.no_grad():
        if kl_per_lt is None:
            kl_per_lt = {
                "latent_dim": [],
                "kl_divergence": [],
                "num_conds": [],
            }

        if kl_vs_rcl is None:
            kl_vs_rcl = {
                "num_conds": [],
                "KLD": [],
                "RCL": [],
                "ELBO": []
            }
        all_kl, all_lt = [], []

        tmp1, tmp2 = torch.split(d, int(d.size()[-1] / 2), dim=1)
        for kk in conds:
            tmp1[:, kk], tmp2[:, kk] = 0, 0
        cond_d = torch.cat((tmp1, tmp2), 1)

        # Run 10 times for resampling
        my_recon_list, my_z_means_list, my_log_var_list = [], [], []
        for resample in range(10):
            recon_batch, z_means, log_var = model(
                c.clone().cuda(args.gpu_id), cond_d.clone().cuda(args.gpu_id)
            )
            my_recon_list.append(recon_batch)
            my_z_means_list.append(z_means)
            my_log_var_list.append(log_var)

        recon_batch = torch.mean(torch.stack(my_recon_list), dim=0)
        z_means = torch.mean(torch.stack(my_z_means_list), dim=0)
        log_var = torch.mean(torch.stack(my_log_var_list), dim=0)

        loss_fn = str_to_object(args.loss_fn)

        elbo_loss_total, rcl_per_lt_temp_total, kl_per_lt_temp_total, _, _ = loss_fn(
            c.cuda(args.gpu_id),
            recon_batch.cuda(args.gpu_id),
            z_means,
            log_var,
            args
        )
        # print('conds is', conds, elbo_loss_total, rcl_per_lt_temp_total, kl_per_lt_temp_total)
        kl_vs_rcl['num_conds'].append(c.size()[-1] - len(conds))
        kl_vs_rcl['KLD'].append(kl_per_lt_temp_total.item())
        kl_vs_rcl['RCL'].append(rcl_per_lt_temp_total.item())
        kl_vs_rcl['ELBO'].append(elbo_loss_total.item())

        for ii in range(z_means.size()[-1]):
            elbo_loss, rcl_per_lt_temp, kl_per_lt_temp, _, _ = loss_fn(
                c.cuda(args.gpu_id),
                recon_batch.cuda(args.gpu_id),
                z_means[:, ii],
                log_var[:, ii],
                args
            )
            # print(elbo_loss.item(), rcl_per_lt_temp.item(), kl_per_lt_temp.item())

            all_kl = np.append(all_kl, kl_per_lt_temp.item())
            all_lt.append(ii)
            kl_per_lt["num_conds"].append(c.size()[-1] - len(conds))
            kl_per_lt["latent_dim"].append(ii)
            kl_per_lt["kl_divergence"].append(kl_per_lt_temp.item())
        all_kl, all_lt = list(zip(*sorted(zip(all_kl, all_lt))))
        all_kl = list(all_kl)
        all_lt = list(all_lt)

        z_means_x = np.append(z_means_x, z_means[:, all_lt[-1]].data.cpu().numpy())
        z_means_y = np.append(z_means_y, z_means[:, all_lt[-2]].data.cpu().numpy())
        z_var_x = np.append(z_var_x, log_var[:, all_lt[-1]].data.cpu().numpy())
        z_var_y = np.append(z_var_y, log_var[:, all_lt[-2]].data.cpu().numpy())
    return z_means_x, z_means_y, kl_per_lt, z_var_x, z_var_y, kl_vs_rcl


def get_sorted_klds(latent_dict):
    df = pd.DataFrame(latent_dict)
    df = df.sort_values(by=["kl_divergence"])
    n_dim = np.max(df["latent_dim"])

    kld_avg_dim = np.zeros(n_dim)

    for i in range(n_dim):
        kld_avg_dim[i] = np.mean(df["kl_divergence"][df["latent_dim"] == i])
    kld_avg_dim = np.sort(kld_avg_dim)[::-1]

    return kld_avg_dim
