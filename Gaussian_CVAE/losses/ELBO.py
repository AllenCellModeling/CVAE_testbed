import torch
import torch.nn.functional as F

def calculate_loss(x, reconstructed_x, mean, log_var):
        # reconstruction loss
    RCL = F.binary_cross_entropy(reconstructed_x, x, size_average=False)
                # kl divergence loss
    KLD = -0.5 * torch.sum(1 + log_var - mean.pow(2) - log_var.exp())
                        
    return RCL + KLD, RCL, KLD

def synthetic_loss(x, reconstructed_x, mean, log_var):
    """
    MSE loss for reconstruction, KLD loss as per VAE. Also want to output dimension (element) wise RCL and KLD
    """

    loss = torch.nn.MSELoss(size_average=True)
    loss_per_element = torch.nn.MSELoss(size_average=False, reduce = False)

    RCL = loss(reconstructed_x, x)
    RCL_per_element = loss_per_element(reconstructed_x, x)
                # kl divergence loss
    
    KLD = -0.5 * torch.sum(1 + log_var - mean.pow(2) - log_var.exp())
    # print('inside loss')
    # print(mean.size(), log_var.size(), reconstructed_x.size())
    KLD_per_element = -0.5 * (1 + log_var - mean.pow(2) - log_var.exp())
    # print(KLD_per_element.size(),RCL_per_element.size(), x.size())

    return RCL + KLD, RCL, KLD, RCL_per_element, KLD_per_element