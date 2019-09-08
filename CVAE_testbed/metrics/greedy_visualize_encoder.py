import torch
import numpy as np
from CVAE_testbed.utils import str_to_object
import pandas as pd
import itertools


def per_condition_loss(d, c, conds, model, args, idx=1):
    tmp1, tmp2 = torch.split(d, int(d.size()[-1] / 2), dim=1)

    condition_tensor = d.clone()
    tmp1, tmp2 = torch.split(condition_tensor, int(condition_tensor.size()[-1] / 2), dim=1)
    for kk in conds:
        tmp1[:, kk], tmp2[:, kk] = 0, 0
    cond_d = torch.cat((tmp1, tmp2), 1)

    # Run 10 times for resampling
    my_recon_list, my_z_means_list, my_log_var_list = [], [], []
    for resample in range(10):
        recon_batch, z_means, log_var = model(
            c.cuda(args.gpu_id), cond_d.cuda(args.gpu_id)
        )
        my_recon_list.append(recon_batch)
        my_z_means_list.append(z_means)
        my_log_var_list.append(log_var)

    recon_batch = torch.mean(torch.stack(my_recon_list), dim=0)
    z_means = torch.mean(torch.stack(my_z_means_list), dim=0)
    log_var = torch.mean(torch.stack(my_log_var_list), dim=0)
    loss_fn = str_to_object(args.loss_fn)

    ELBO_loss, RCL_loss, KLD_loss, _, _ = loss_fn(
        c.cuda(args.gpu_id),
        recon_batch.cuda(args.gpu_id),
        z_means,
        log_var, 
        args
    )
    if idx == 1:
        return ELBO_loss.item()
    else:
        return recon_batch, z_means, log_var

def GreedyVisualizeEncoder(args, model, conds, c, d, kl_per_lt=None, kl_all_lt=None):
    """
    In this, we just take num_of_conds and run all combinations
    so if num_of_conds is 10, then we run all combinations of 10 features from the total list of 159 features
    """

    z_means_x, z_means_y = [], []
    z_var_x, z_var_y = [], []
    model.cuda(args.gpu_id)
    all_conds = [i for i in range(c.size()[-1])]

    with torch.no_grad():
        if kl_per_lt is None:
            kl_per_lt = {
                "latent_dim": [],
                "kl_divergence": [],
                "num_conds": [],
                "popped_cond": [],
                "rcl": [],
                "elbo": [],
                "cond_comb": []
            }
        if kl_all_lt is None:
            kl_all_lt = {
                "z_means_x": [],
                "z_var_x": [],
                "z_var_y": [],
                "num_conds": [],
                "z_means_y": [],
                "cond_comb": []
            }

        this_conditions_ELBO = per_condition_loss(d.clone(), c.clone(), conds.copy(), model, args)
        if len(conds) == c.size()[-1]:
            recon_batch, z_means, log_var = per_condition_loss(d.clone(), c.clone(), conds.copy(), model, args, idx = 2)
            kl_per_lt, kl_all_lt = make_dataframe(kl_per_lt, kl_all_lt, z_means, c, recon_batch, log_var, conds.copy(), [], args)

        max_ELBO_diff = 0
        most_important_cond = []
        popped_cond = 0

        for j, i in enumerate(range(len(conds))):
            print(i)
            this_conds = conds.copy()
            del this_conds[i]
            print(this_conds)
            this_conditions_ELBO_plus_one = per_condition_loss(d.clone(), c.clone(), this_conds.copy(), model, args)
            ELBO_diff = np.abs(this_conditions_ELBO - this_conditions_ELBO_plus_one)
            print(ELBO_diff)
            if ELBO_diff > max_ELBO_diff:
                max_ELBO_diff = ELBO_diff
                most_important_cond = this_conds
                popped_cond = i
        print(most_important_cond, max_ELBO_diff)
        recon_batch, z_means, log_var = per_condition_loss(d.clone(), c.clone(), most_important_cond.copy(), model, args, idx = 2)
        
        kl_per_lt, kl_all_lt = make_dataframe(kl_per_lt, kl_all_lt, z_means, c, recon_batch, log_var, most_important_cond, popped_cond, args)

        if len(most_important_cond) != 0:
            kl_per_lt, kl_all_lt = GreedyVisualizeEncoder(args, model, most_important_cond, c, d, kl_per_lt, kl_all_lt)
        else:
            # recon_batch, z_means, log_var = per_condition_loss(d.clone(), c.clone(), most_important_cond.copy(), model, args, idx = 2)
            # kl_per_lt, kl_all_lt = make_dataframe(kl_per_lt, kl_all_lt, z_means, c, recon_batch, log_var, most_important_cond, [], args)
            
            return kl_per_lt, kl_all_lt
    return kl_per_lt, kl_all_lt


def make_dataframe(kl_per_lt, kl_all_lt, z_means, c, recon_batch, log_var, conds, popped_cond, args):
    
    z_means_x, z_var_x, z_means_y, z_var_y = [], [], [], []
    all_kl, all_lt = [], []

    for ii in range(z_means.size()[-1]):
        loss_fn = str_to_object(args.loss_fn)

        _, rcl_per_lt_temp, kl_per_lt_temp, _, _ = loss_fn(
            c.cuda(args.gpu_id),
            recon_batch.cuda(args.gpu_id),
            z_means[:, ii],
            log_var[:, ii], 
            args
        )

        all_kl = np.append(all_kl, kl_per_lt_temp.item())
        all_lt.append(ii)
        kl_per_lt["num_conds"].append(c.size()[-1] - len(conds))
        kl_per_lt["popped_cond"].append(popped_cond)
        kl_per_lt["latent_dim"].append(ii)
        kl_per_lt["kl_divergence"].append(kl_per_lt_temp.item())
        kl_per_lt["rcl"].append(rcl_per_lt_temp.item())
        kl_per_lt["elbo"].append(rcl_per_lt_temp.item() + kl_per_lt_temp.item())
        kl_per_lt['cond_comb'].append(str([i for i in conds]))
    all_kl, all_lt = list(zip(*sorted(zip(all_kl, all_lt))))
    all_kl = list(all_kl)
    all_lt = list(all_lt)

    z_means_x = np.append(z_means_x, z_means[:, all_lt[-1]].data.cpu().numpy())
    z_means_y = np.append(z_means_y, z_means[:, all_lt[-2]].data.cpu().numpy())
    z_var_x = np.append(z_var_x, log_var[:, all_lt[-1]].data.cpu().numpy())
    z_var_y = np.append(z_var_y, log_var[:, all_lt[-2]].data.cpu().numpy())
    kl_all_lt['z_means_x'].append(z_means_x)
    kl_all_lt['z_means_y'].append(z_means_y)
    kl_all_lt['z_var_x'].append(z_var_x)
    kl_all_lt['z_var_y'].append(z_var_y)
    kl_all_lt['num_conds'].append(c.size()[-1] - len(conds))
    kl_all_lt['cond_comb'].append(str([i for i in conds]))

    return kl_per_lt, kl_all_lt


def get_sorted_klds(latent_dict):
    df = pd.DataFrame(latent_dict)
    df = df.sort_values(by=["kl_divergence"])
    n_dim = np.max(df["latent_dim"])

    kld_avg_dim = np.zeros(n_dim)

    for i in range(n_dim):
        kld_avg_dim[i] = np.mean(df["kl_divergence"][df["latent_dim"] == i])
    kld_avg_dim = np.sort(kld_avg_dim)[::-1]

    return kld_avg_dim
