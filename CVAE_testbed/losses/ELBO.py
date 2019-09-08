import torch
import torch.nn.functional as F
from torch.distributions.log_normal import LogNormal


def calculate_loss(x, reconstructed_x, mean, log_var):
    # reconstruction loss
    RCL = F.binary_cross_entropy(reconstructed_x, x, size_average=False)
    # kl divergence loss
    KLD = -0.5 * torch.sum(1 + log_var - mean.pow(2) - log_var.exp())        
    return RCL + KLD, RCL, KLD


def synthetic_loss(x, reconstructed_x, mean, log_var, args):
    """
    MSE loss for reconstruction, 
    KLD loss as per VAE. 
    Also want to output dimension (element) wise RCL and KLD
    """
    # print(x.size(), mean.size(), log_var.size())
    loss = torch.nn.MSELoss(size_average=False)
    loss_per_element = torch.nn.MSELoss(
        size_average=False, reduce=False)

    # if x is masked, find indices in x
    # that are 0 and set reconstructed x to 0 as well
    indices = x == 0
    reconstructed_x[indices] = 0

    RCL = loss(reconstructed_x, x)
    RCL_per_element = loss_per_element(reconstructed_x, x)
    KLD = -0.5 * torch.sum(1 + log_var - mean.pow(2) - log_var.exp())
    KLD_per_element = -0.5 * (1 + log_var - mean.pow(2) - log_var.exp())

    return RCL + args.beta_vae*KLD, RCL, KLD, RCL_per_element, KLD_per_element


def synthetic_loss_no_mask(x, reconstructed_x, mean, log_var, args):
    """
    MSE loss for reconstruction, KLD loss as per VAE. 
    Also want to output dimension (element) wise RCL and KLD
    """

    loss = torch.nn.MSELoss(size_average=False)
    loss_per_element = torch.nn.MSELoss(size_average=False, reduce=False)

    RCL = loss(reconstructed_x, x)
    RCL_per_element = loss_per_element(reconstructed_x, x)
    KLD = -0.5 * torch.sum(1 + log_var - mean.pow(2) - log_var.exp())
    KLD_per_element = -0.5 * (1 + log_var - mean.pow(2) - log_var.exp())

    return RCL + args.beta_vae*KLD, RCL, KLD, RCL_per_element, KLD_per_element


def combined_loss(x, reconstructed_x, mean, log_var, args):
    """
    MSE loss for reconstruction, KLD loss as per VAE.
    Also want to output dimension (element) wise RCL and KLD
    """
    # First, binary data
    loss1 = torch.nn.BCEWithLogitsLoss(size_average=False)
    loss1_per_element = torch.nn.BCEWithLogitsLoss(
                                                    size_average=False,
                                                    reduce=False
                                                  )
    binary_range = args.binary_real_one_hot_parameters['binary_range']
    reconstructed_x1 = reconstructed_x[:, binary_range[0]: binary_range[1]]
 
    x1 = x[:, binary_range[0]: binary_range[1]]
    RCL1 = loss1(reconstructed_x1, x1)
    RCL1_per_element = loss1_per_element(reconstructed_x1, x1)

    # Next, real data
    loss2 = torch.nn.MSELoss(size_average=False)
    loss2_per_element = torch.nn.MSELoss(size_average=False, reduce=False)
    real_range = args.binary_real_one_hot_parameters['real_range']
    reconstructed_x2 = reconstructed_x[:, real_range[0]: real_range[1]]
    x2 = x[:, real_range[0]: real_range[1]]

    RCL2 = loss2(reconstructed_x2, x2)
    RCL2_per_element = loss2_per_element(reconstructed_x2, x2)

    # Next, one-hot data
    loss3 = torch.nn.CrossEntropyLoss(size_average=True)
    loss3_per_element = torch.nn.CrossEntropyLoss(
                                                    size_average=True,
                                                    reduce=False
                                                 )

    one_hot_range = args.binary_real_one_hot_parameters['one_hot_range']
    reconstructed_x3 = reconstructed_x[:, one_hot_range[0]: one_hot_range[1]]
    x3 = x[:, one_hot_range[0]: one_hot_range[1]]
    # This has 3 one-hot's. lets split it up

    x3_1 = x3[:, :19]
    x3_2 = x3[:, 19:19 + 19]
    x3_3 = x3[:, 19+19:]
    reconstructed_x3_1 = reconstructed_x3[:, :19]
    reconstructed_x3_2 = reconstructed_x3[:, 19:19 + 19]
    reconstructed_x3_3 = reconstructed_x3[:, 19+19:]

    _, labels1 = x3_1.max(dim=1)
    _, labels2 = x3_2.max(dim=1)
    _, labels3 = x3_3.max(dim=1)
    # print(labels.size(), reconstructed_x3.size(), x3.size())
    RCL3_1 = loss3(reconstructed_x3_1, labels1.long())
    RCL3_per_element_1 = loss3_per_element(reconstructed_x3_1, labels1.long())
    RCL3_2 = loss3(reconstructed_x3_2, labels2.long())
    RCL3_per_element_2 = loss3_per_element(reconstructed_x3_2, labels2.long())
    RCL3_3 = loss3(reconstructed_x3_3, labels3.long())
    RCL3_per_element_3 = loss3_per_element(reconstructed_x3_3, labels3.long())

    KLD = -0.5 * torch.sum(1 + log_var - mean.pow(2) - log_var.exp())
    KLD_per_element = -0.5 * (1 + log_var - mean.pow(2) - log_var.exp())
    RCL = RCL1 + RCL2 + RCL3_1 + RCL3_2 + RCL3_3

    RCL_per_element = torch.cat(
        (
            RCL1_per_element,
            RCL2_per_element,
            RCL3_per_element_1.view([-1, 1]),
            RCL3_per_element_2.view([-1, 1]),
            RCL3_per_element_3.view([-1, 1])
        ),
        1
                               )

    return RCL + args.beta_vae*KLD, RCL, KLD, RCL_per_element, KLD_per_element


def synthetic_loss_baseline_2(
    z0,
    z0_logits,
    z1,
    z1_prior,
    z2,
    z1_post,
    z2_post,
    args
                            ):
    """
    Based on encoding a single bit repo
    MSE loss for reconstruction, KLD loss as per VAE. 
    Also want to output dimension (element) wise RCL and KLD
    """

    loss = torch.nn.BCEWithLogitsLoss(size_average=True)
    loss_per_element = torch.nn.BCEWithLogitsLoss(size_average=True, reduce=False)

    # First RCL term
    RCL = loss(z0_logits, z0)
    RCL_per_element = loss_per_element(z0_logits, z0)

    # First KL term
    print(z1_prior[0].size(), z1_post[0].size())
    # print(LogNormal(z1_prior[0], z1_prior[1]).entropy())
    KLD1 = torch.sum(-LogNormal(*z1_prior).entropy() + LogNormal(*z1_post).entropy())

    KLD2 = torch.sum(-LogNormal(0, 1).entropy() + LogNormal(*z2_post).entropy())

    KLD_per_element = -0.5 * (1 + z1_post[1] - z1_post[0].pow(2) - z1_post[1].exp())

    print(KLD1, KLD2, RCL)
    KLD = KLD1 + KLD2

    return RCL + args.beta_vae*KLD, RCL, KLD, RCL_per_element, KLD_per_element
