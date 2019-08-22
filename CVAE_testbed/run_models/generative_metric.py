import torch
import numpy as np
import sys
import matplotlib.pyplot as plt
import torchvision
# sys.path.insert(0, '../models/')
from CVAE_testbed.models.CVAE_first import idx2onehot
# sys.path.insert(0, '../metrics/')
from CVAE_testbed.metrics.inception import InceptionV3
from CVAE_testbed.metrics.calculate_fid import get_activations, calculate_frechet_distance


def compute_generative_metric(test_iterator, model, device, LATENT_DIM, BATCH_SIZE, color_value=None, digit_value=None):
    inc = InceptionV3([3])
    inc = inc.cuda()
    
    with torch.no_grad():
        im = torch.empty([0])
        lab = torch.empty([0])
        for imm, tll in iter(test_iterator):
            for tim, tl in zip(imm, tll):
                if lab.size()[0] != 500:
                    if digit_value is not None:
                        if tl == digit_value:
                            im = torch.cat((im, tim), 0)
                            lab = torch.cat((lab, tl.view(1).float()), 0)
                    else:
                        im = torch.cat((im, tim), 0)
                        lab = torch.cat((lab, tl.view(1).float()), 0)
                elif lab.size()[0] == 500:
                    break
        im = im.view(lab.size()[0], 1, 28, 28)
        im = im.repeat(1,3,1,1)

        colors = []

        for j in range(lab.size()[0]):
            if color_value is not None:
                color = torch.randint(color_value+ 1, color_value+2, (1,1)).item()
            else:
                color = torch.randint(1, 4, (1,1)).item()
            other_indices = []
            color_index = []
            for a in [1, 2, 3]:
                if color != a:
                    other_indices.append(a)
                else:
                    other_index = a
            im[j, other_indices[0]-1, :, :].fill_(0)
            im[j, other_indices[1]-1, :, :].fill_(0)
            colors.append(color - 1)

        colors = torch.FloatTensor(colors)

        z = torch.randn(lab.size()[0], LATENT_DIM).to(device)

        if digit_value is not None:
            y = torch.randint(digit_value,digit_value + 1, (lab.size()[0], 1)).to(dtype=torch.long)
        else:
            y = torch.randint(0,10, (lab.size()[0], 1)).to(dtype=torch.long)

        y = idx2onehot(y, n = 10).to(device, dtype=z.dtype)

        y = torch.cat((y, torch.zeros([lab.size()[0]]).view(-1, 1).cuda()), dim = 1)

        if color_value is not None:
            y2 = torch.randint(color_value, color_value+1, (lab.size()[0], 1)).to(dtype=torch.long)
        else:
            y2 = torch.randint(0, 3, (lab.size()[0], 1)).to(dtype = torch.long)

        y2 = idx2onehot(y2, n = 3).to(device, dtype=z.dtype)
        y2 = torch.cat((y2, torch.zeros([lab.size()[0]]).view(-1, 1).cuda()), dim = 1)
        y = torch.cat((y, y2), dim = 1)

        z = torch.cat((z, y), dim = 1)

        generated_x = model.decoder(z, y)

        X_act = get_activations(im.cpu().data.numpy(), inc, batch_size=BATCH_SIZE, dims=2048, cuda=True)
        recon_act = get_activations(generated_x.cpu().data.numpy(), inc,batch_size=BATCH_SIZE, dims=2048, cuda=True)

        X_act_mu = np.mean(X_act, axis=0)
        recon_act_mu = np.mean(recon_act, axis=0)
        X_act_sigma = np.cov(X_act, rowvar=False)
        recon_act_sigma = np.cov(recon_act, rowvar=False)

        fid = calculate_frechet_distance(X_act_mu, X_act_sigma, recon_act_mu,recon_act_sigma, eps=1e-6)

        images = im[:5, :, :, :]
        gen_images = generated_x[:5, :, :, :]

        grid = torchvision.utils.make_grid(images, nrow = 5)
       
        grid2 = torchvision.utils.make_grid(gen_images, nrow=5)

    return fid, grid.cpu(), grid2.cpu()
