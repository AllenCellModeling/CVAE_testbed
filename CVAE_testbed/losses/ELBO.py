import torch
import torch.nn.functional as F


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