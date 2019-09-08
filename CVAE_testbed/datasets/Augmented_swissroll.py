import torch
from torch.utils.data import DataLoader, Dataset
from torch.distributions import MultivariateNormal
import numpy as np
from sklearn import manifold, datasets

class SwissRoll(Dataset):
    def __init__(self, num_batches, BATCH_SIZE, model_kwargs, shuffle=True, corr=False, train=True, mask=False):
        """
        Args: 
            num_batches: Number of batches of synthetic data
            BATCH_SIZE: batchsize of synthetic data
            model_kwargs: dictionary containing "x_dim" which indicates input data size
            shuffle:  True sets condition vector in input data to 0 for all possible permutations
            corr: True sets dependent input dimensions via a correlation matrix 
        """
        self.num_batches = num_batches
        self.BATCH_SIZE = BATCH_SIZE
        self.corr = corr
        self.shuffle = shuffle
        self.model_kwargs = model_kwargs
        self.train = train

        Batches_X, Batches_C, Batches_conds = torch.empty([0]) ,torch.empty([0]), torch.empty([0])

        for j, i in enumerate(range(self.num_batches)):

            # set parameters
            length_phi = 15   #length of swiss roll in angular direction
            length_Z = 15     #length of swiss roll in z direction
            sigma = 0.1       #noise strength
            m = self.BATCH_SIZE         #number of samples

            # create dataset
            phi = length_phi*np.random.rand(m)
            xi = np.random.rand(m)
            Z = length_Z*np.random.rand(m)
            X = 1./6*(phi + sigma*xi)*np.sin(phi)
            Y = 1./6*(phi + sigma*xi)*np.cos(phi)

            swiss_roll = torch.from_numpy(np.array([X, Y, Z]).transpose()).float()
            self._color = np.sqrt(X**2 + Y**2)
            radius = torch.from_numpy(self._color).view(-1, 1).float()

            swiss_roll = torch.cat((swiss_roll, radius), 1)
            if mask is True:
                mask_indices = torch.cuda.FloatTensor(swiss_roll.size()[0]).uniform_() > 1 - model_kwargs['mask_percentage']
                swiss_roll[mask_indices, 0] = 0
                swiss_roll[mask_indices, 1] = 0
                swiss_roll[mask_indices, 2] = 0
                swiss_roll[mask_indices, 3] = 0
        
            C = swiss_roll.clone()
            count = 0
            if self.shuffle is True:
                while count == 0:
                    C_mask = torch.zeros(C.shape).bernoulli_(0.5)
                    # 3 here refers to 3 dimensions in swiss roll
                    if len(set([i.item() for i in torch.sum(C_mask, dim = 1)])) == 4 + 1:
                        count = 1 
            else:
                C_mask = torch.zeros(C.shape).bernoulli_(0)

            C[C_mask.byte()] = 0
            C_indicator = C_mask == 0

            C = torch.cat([C.float(), C_indicator.float()], 1)

            # 4 here is number of dimensions in swiss roll
            swiss_roll = swiss_roll.view([1, -1, 4])
            C = C.view([1, -1, 4*2])

            # Sum up
            conds = C[:,:,4:].sum(2)
            Batches_X = torch.cat([Batches_X, swiss_roll], 0)
            Batches_C = torch.cat([Batches_C, C], 0)
            Batches_conds = torch.cat([Batches_conds, conds], 0)

        self._batches_x = Batches_X
        self._batches_c = Batches_C
        self._batches_conds = Batches_conds
    
    def __len__(self):
        return len(self._batches_x)

    def __getitem__(self, idx):
        """
        Returns a tuple. (X, C, sum(C[mid:end])). 
        X is the input, 
        C is the condition, 
        sum(C[mid:end]) is the sum of the indicators in C. It tells us how many of the condition
        columns have been masked
        """
        return self._batches_x[idx], self._batches_c[idx], self._batches_conds[idx]

    def get_all_items(self):
        if self.train is True:
            return self._batches_x, self._batches_c, self._batches_conds, None
        else:
            return self._batches_x, self._batches_c, self._batches_conds

    def get_color(self):
        return self._color


