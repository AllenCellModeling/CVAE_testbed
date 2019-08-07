import torch
import torch.nn.functional as F

def calculate_loss(x, reconstructed_x, mean, log_var):
        # reconstruction loss
    RCL = F.binary_cross_entropy(reconstructed_x, x, size_average=False)
                # kl divergence loss
    KLD = -0.5 * torch.sum(1 + log_var - mean.pow(2) - log_var.exp())
                        
    return RCL + KLD, RCL, KLD

def synthetic_loss(x, reconstructed_x, mean, log_var):
        # reconstruction loss
    #RCL = F.binary_cross_entropy(reconstructed_x, x, size_average=False)
    loss = torch.nn.MSELoss()
    # loss = torch.nn.SmoothL1Loss()
#     RCL = nn.MSELoss(reconstructed_x, x)
    RCL = loss(reconstructed_x, x)
                # kl divergence loss
    KLD = -0.5 * torch.sum(1 + log_var - mean.pow(2) - log_var.exp())
    return RCL + KLD, RCL, KLD