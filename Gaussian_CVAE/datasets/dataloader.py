import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import torchvision
from torch.distributions import MultivariateNormal

def load_mnist_data(BATCH_SIZE, model_kwargs):
    from torchvision import transforms
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    transforms = transforms.Compose([transforms.ToTensor()])

    train_dataset = datasets.MNIST(
    './datasets/mnist_data',
    train=True,
    download=True,
    transform=transforms)

    test_dataset = datasets.MNIST(
        './datasets/mnist_data',
        train=False,
        download=True,
        transform=transforms
    )

    train_iterator = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    test_iterator = DataLoader(test_dataset, batch_size=BATCH_SIZE)

    return train_iterator, test_iterator

def make_synthetic_data(num_batches, BATCH_SIZE, model_kwargs, shuffle=True):

    Batches_X, Batches_C, Batches_conds = torch.empty([0]) ,torch.empty([0]), torch.empty([0])

    for i in range(num_batches):
        m = MultivariateNormal(torch.zeros(model_kwargs['x_dim']), torch.eye(model_kwargs['x_dim']))
        X = m.sample((BATCH_SIZE,))

        C = X.clone()
        count = 0
        if shuffle is True:
            while count == 0:
                C_mask = torch.zeros(C.shape).bernoulli_(0.5)
                if len(set([i.item() for i in torch.sum(C_mask, dim = 1)])) == model_kwargs['x_dim'] + 1:
                    count = 1 
        else:
            C_mask = torch.zeros(C.shape).bernoulli_(0)
        print(len(set([i.item() for i in torch.sum(C_mask, dim = 1)])))

        C[C_mask.byte()] = 0
        C_indicator = C_mask == 0

        C = torch.cat([C.float(), C_indicator.float()], 1)
        X = X.view([1, -1, model_kwargs['x_dim']])
        C = C.view([1, -1, model_kwargs['x_dim']*2])

        # Sum up
        conds = C[:,:,model_kwargs['x_dim']:].sum(2)

        Batches_X = torch.cat([Batches_X, X], 0)
        Batches_C = torch.cat([Batches_C, C], 0)
        Batches_conds = torch.cat([Batches_conds, conds], 0)

    return Batches_X, Batches_C, Batches_conds
