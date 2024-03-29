import torch
from torch.utils.data import DataLoader, Dataset
from torch.distributions import MultivariateNormal
import numpy as np
from sklearn.decomposition import PCA


class SyntheticDataset(Dataset):
    def __init__(
                self,
                num_batches,
                BATCH_SIZE,
                model_kwargs,
                shuffle=True,
                corr=False,
                train=True,
                mask=False
                ):
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

        Batches_X, Batches_C, Batches_conds = torch.empty([0]), torch.empty([0]), torch.empty([0])
        for j, i in enumerate(range(self.num_batches)):
            if self.corr is False:
                m = MultivariateNormal(torch.zeros(self.model_kwargs['x_dim']), torch.eye(self.model_kwargs['x_dim']))
            else:
                if j == 0:
                    corr_matrix = self.random_corr_mat(D=self.model_kwargs['x_dim'])
                    corr_matrix = torch.from_numpy(corr_matrix)
                m = MultivariateNormal(torch.zeros(self.model_kwargs['x_dim']).float(), corr_matrix.float())

            X = m.sample((self.BATCH_SIZE,))

            C = X.clone()
            count = 0
            if self.shuffle is True:
                while count == 0:
                    C_mask = torch.zeros(C.shape).bernoulli_(0.5)
                    count = 1
                    # if len(set([i.item() for i in torch.sum(C_mask, dim = 1)])) == self.model_kwargs['x_dim'] + 1:
                    #     count = 1 
            else:
                C_mask = torch.zeros(C.shape).bernoulli_(0)

            C[C_mask.byte()] = 0
            C_indicator = C_mask == 0

            C = torch.cat([C.float(), C_indicator.float()], 1)
            X = X.view([1, -1, self.model_kwargs['x_dim']])
            C = C.view([1, -1, self.model_kwargs['x_dim']*2])

            # Sum up
            conds = C[:, :, self.model_kwargs['x_dim']:].sum(2)

            Batches_X = torch.cat([Batches_X, X], 0)
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
        return None

    def random_corr_mat(self, D=10, beta=1):
        """Generate random valid correlation matrix of dimension D.
        Smaller beta gives larger off diagonal correlations (beta > 0)."""

        P = np.zeros([D, D])
        S = np.eye(D)

        for k in range(0, D - 1):
            for i in range(k + 1, D):
                P[k, i] = 2 * np.random.beta(beta, beta) - 1
                p = P[k, i]
                for l in reversed(range(k)):
                    p = (
                        p * np.sqrt((1 - P[l, i] ** 2) * (1 - P[l, k] ** 2)) + P[l, i] * P[l, k]
                    )
                S[k, i] = S[i, k] = p

        p = np.random.permutation(D)
        for i in range(D):
            S[:, i] = S[p, i]
        for i in range(D):
            S[i, :] = S[i, p]
        return S
