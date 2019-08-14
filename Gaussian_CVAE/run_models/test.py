import sys
# sys.path.insert(0, '../losses/')
from Gaussian_CVAE.losses.ELBO import calculate_loss
# sys.path.insert(0, '../models/')
from Gaussian_CVAE.models.sample import Sample
import torch
# sys.path.insert(0, '../metrics/')
from Gaussian_CVAE.metrics.blur import calc_blur


def test(model, optimizer, test_iterator, device, BATCH_SIZE):
    model.eval()
    test_loss = 0
    kld_loss = 0
    rcl_loss = 0
    kl_per_lt = {'Latent_Dimension': [], 'KL_Divergence': [], 'Latent_Mean': [], 'Latent_Variance': []}
    with torch.no_grad():
        for i, (x, y) in enumerate(test_iterator):
            model.to(device)
            sm = Sample(x, y, BATCH_SIZE, device)
            x, y = sm.generate_x_y()
            x = x.view(-1,3,28,28)

            reconstructed_x, z_mu, z_var, _ = model(x, y)
            blur = calc_blur(reconstructed_x)

            for ii in range(z_mu.size()[-1]) :
                 _, _, kl_per_lt_temp = calculate_loss(x, reconstructed_x, z_mu[:, ii], z_var[:, ii])
                 kl_per_lt['Latent_Dimension'].append(ii)
                 kl_per_lt['Latent_Mean'].append(z_mu[:, ii])
                 kl_per_lt['Latent_Variance'].append(z_var[:, ii])
            loss, rcl, kld = calculate_loss(x, reconstructed_x, z_mu, z_var)
            test_loss += loss.item()
            rcl_loss += rcl.item()
            kld_loss += kld.item()
    return test_loss, rcl_loss, kld_loss, kl_per_lt, blur.data.item()
