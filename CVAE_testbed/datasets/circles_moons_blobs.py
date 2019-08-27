import torch
from torch.utils.data import DataLoader, Dataset
from sklearn import manifold, datasets


class CirclesMoonsBlobs(Dataset):
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

        Batches_X, Batches_C, Batches_conds = torch.empty([0]), torch.empty([0]), torch.empty([0])

        for j, i in enumerate(range(self.num_batches)):
            m = self.BATCH_SIZE

            # create dataset
            if model_kwargs['sklearn_data'] == 'circles':
                X = torch.from_numpy(
                    datasets.make_circles(n_samples=m, factor=.5, noise=.05)[0]
                    ).float()
            elif model_kwargs['sklearn_data'] == 'blobs':
                X = torch.from_numpy(
                    datasets.make_blobs(n_samples=m, random_state=8)[0]
                    ).float()
            elif model_kwargs['sklearn_data'] == 'moons':
                X = torch.from_numpy(datasets.make_moons(n_samples=m, noise=.05)[0]).float()
            elif model_kwargs['sklearn_data'] == 's_curve':
                X = torch.from_numpy(datasets.make_s_curve(n_samples=m, noise=0.05)[0]).float()
            else:
                break

            self._color = X[:, 0]

            if mask is True:
                mask_indices = torch.cuda.FloatTensor(X.size()[0]).uniform_() > 1 - model_kwargs['mask_percentage']
                X[mask_indices, 0] = 0
                X[mask_indices, 1] = 0
        
            C = X.clone()
            count = 0
            if self.shuffle is True:
                while count == 0:
                    C_mask = torch.zeros(C.shape).bernoulli_(0.5)
                    # 3 here refers to 3 dimensions in swiss roll
                    if len(set([i.item() for i in torch.sum(C_mask, dim=1)])) == X.size()[-1] + 1:
                        count=1 
            else:
                C_mask = torch.zeros(C.shape).bernoulli_(0)

            C[C_mask.byte()] = 0
            C_indicator = C_mask == 0

            C = torch.cat([C.float(), C_indicator.float()], 1)

            # 4 here is number of dimensions in swiss roll
            X = X.view([1, -1, X.size()[-1]])
            C = C.view([1, -1, X.size()[-1]*2])

            # Sum up
            conds = C[:, :, X.size()[-1]:].sum(2)
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
            return self._batches_x, self._batches_c, self._batches_conds, self._P
        else:
            return self._batches_x, self._batches_c, self._batches_conds
    
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

    def get_color(self):
        return self._color

    def get_all_items(self):
        if self.train is True:
            return self._batches_x, self._batches_c, self._batches_conds, None
        else:
            return self._batches_x, self._batches_c, self._batches_conds

