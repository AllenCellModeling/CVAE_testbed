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
            c.clone().cuda(args.gpu_id), cond_d.clone().cuda(args.gpu_id)
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
    elif idx == 5:
        return ELBO_loss.item(), RCL_loss.item(), KLD_loss.item()
    else:
        return recon_batch, z_means, log_var

def GreedyVisualizeEncoder(args, model, conds, c, d, kl_per_lt=None, kl_all_lt=None, selected_features=None, feature_names=None, first_features=None):
    """
    In this, we just take num_of_conds and run all combinations
    so if num_of_conds is 10, then we run all combinations of 10 features from the total list of 159 features
    """

    model.cuda(args.gpu_id)

    with torch.no_grad():
        if kl_per_lt is None:
            kl_per_lt = {
                "latent_dim": [],
                "kl_divergence": [],
                "num_conds": [],
                "popped_cond": [],
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
        
        if selected_features is None:
            selected_features = {
                "selected_feature_number": [],
                "selected_feature_name": [],
                "ELBO": [],
                "RCL": [],
                "KLD": []
            }
        
        if first_features is None:
            first_features = {
                "selected_feature_number": [],
                "selected_feature_name": [],
                "ELBO": [],
                "RCL": [],
                "KLD": []
            }

        this_conditions_ELBO = per_condition_loss(d.clone(), c.clone(), conds.copy(), model, args)

        max_num_conds = c.size()[-1]
        # print(c.size()[-1], 'first')
        if len(conds) == c.size()[-1]:
            recon_batch, z_means, log_var = per_condition_loss(d.clone(), c.clone(), conds.copy(), model, args, idx = 2)
            kl_per_lt, kl_all_lt, selected_features = make_dataframe(kl_per_lt, kl_all_lt, selected_features, feature_names, z_means, c.clone(), recon_batch, log_var, conds.copy(), [], args)

        max_ELBO_diff = 0
        most_important_cond = []
        popped_cond = 0

        for j, i in enumerate(range(len(conds))):
            # print(i)
            this_conds = conds.copy()
            del this_conds[i]
            # print(this_conds)
            if len(this_conds) == max_num_conds - 1:
                # print('inside')
                this_conditions_ELBO_plus_one, RCL_one_cond, KLD_one_cond = per_condition_loss(d.clone(), c.clone(), this_conds.copy(), model, args, idx=5)
                # print(this_conditions_ELBO_plus_one)
                first_features['selected_feature_number'].append(str(conds[i]))
                if feature_names is not None:
                    first_features['selected_feature_name'].append(feature_names[i])
                else:
                    first_features['selected_feature_name'].append(None)
                first_features['ELBO'].append(this_conditions_ELBO_plus_one)
                first_features['RCL'].append(RCL_one_cond)
                first_features['KLD'].append(KLD_one_cond)
            else:
                this_conditions_ELBO_plus_one = per_condition_loss(d.clone(), c.clone(), this_conds.copy(), model, args)
            ELBO_diff = this_conditions_ELBO - this_conditions_ELBO_plus_one
            # print(ELBO_diff)
            if ELBO_diff > max_ELBO_diff:
                max_ELBO_diff = ELBO_diff
                most_important_cond = this_conds
                popped_cond = conds[i]
        # print('after loop', most_important_cond, popped_cond)
        recon_batch, z_means, log_var = per_condition_loss(d.clone(), c.clone(), most_important_cond.copy(), model, args, idx = 2)
        
        kl_per_lt, kl_all_lt, selected_features = make_dataframe(kl_per_lt, kl_all_lt, selected_features, feature_names, z_means, c.clone(), recon_batch, log_var, most_important_cond, popped_cond, args)

        if len(most_important_cond) != 0:
            # print('first features', first_features)
            kl_per_lt, kl_all_lt, selected_features, first_features = GreedyVisualizeEncoder(args, model, most_important_cond, c.clone(), d.clone(), kl_per_lt, kl_all_lt, selected_features, feature_names, first_features)
        else:
            # recon_batch, z_means, log_var = per_condition_loss(d.clone(), c.clone(), most_important_cond.copy(), model, args, idx = 2)
            # kl_per_lt, kl_all_lt = make_dataframe(kl_per_lt, kl_all_lt, z_means, c, recon_batch, log_var, most_important_cond, [], args)
            return kl_per_lt, kl_all_lt, selected_features, first_features
    return kl_per_lt, kl_all_lt, selected_features, first_features


def make_dataframe(kl_per_lt, kl_all_lt, selected_features, feature_names, z_means, c, recon_batch, log_var, conds, popped_cond, args):
    
    z_means_x, z_var_x, z_means_y, z_var_y = [], [], [], []
    all_kl, all_lt = [], []

    loss_fn = str_to_object(args.loss_fn)

    total_ELBO, total_RCL, total_KLD, _, _ = loss_fn(
        c.cuda(args.gpu_id),
        recon_batch.cuda(args.gpu_id),
        z_means,
        log_var,
        args
    )
    selected_features["ELBO"].append(total_ELBO.item())
    selected_features["RCL"].append(total_RCL.item())
    selected_features["KLD"].append(total_KLD.item())
    if popped_cond != []:
        selected_features["selected_feature_number"].append(str(popped_cond))
        if args.data_type == 'aics_features':
            name = feature_names[popped_cond]
            selected_features["selected_feature_name"].append(name)
        else:
            selected_features["selected_feature_name"].append(None)
    else:
        selected_features["selected_feature_number"].append(None)
        selected_features["selected_feature_name"].append(None)
    
    for ii in range(z_means.size()[-1]):
        
        _, rcl_per_lt_temp, kl_per_lt_temp, _, _ = loss_fn(
            c.cuda(args.gpu_id),
            recon_batch.cuda(args.gpu_id),
            z_means[:, ii],
            log_var[:, ii],
            args
        )

        all_kl = np.append(all_kl, kl_per_lt_temp.item())
        all_lt.append(ii)
        # print('greedy encoding plots', c.size()[-1] - len(conds))
        kl_per_lt["num_conds"].append(c.size()[-1] - len(conds))
        if popped_cond != []:
            kl_per_lt["popped_cond"].append(popped_cond)
        else:
            kl_per_lt["popped_cond"].append(None)
        kl_per_lt["latent_dim"].append(ii)
        kl_per_lt["kl_divergence"].append(kl_per_lt_temp.item())
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

    return kl_per_lt, kl_all_lt, selected_features


def get_sorted_klds(latent_dict):
    df = pd.DataFrame(latent_dict)
    df = df.sort_values(by=["kl_divergence"])
    n_dim = np.max(df["latent_dim"])

    kld_avg_dim = np.zeros(n_dim)

    for i in range(n_dim):
        kld_avg_dim[i] = np.mean(df["kl_divergence"][df["latent_dim"] == i])
    kld_avg_dim = np.sort(kld_avg_dim)[::-1]

    return kld_avg_dim


