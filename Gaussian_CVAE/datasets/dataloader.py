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

def initialize_synthetic_data(BATCH_SIZE, model_kwargs):


    return input_data

def make_subset_0(row, n_cols):
    indices = []
    for i in range(n_cols):

        ind = torch.randint(0, 2, (1,)).item()
        indices.append(ind)
        row[ind] = 0
    encoded = torch.ones(row.size())
    for i in range(n_cols):
        encoded[indices[i]] = 0
    concat_row = torch.cat((row.view(1, -1), encoded.view(1, -1)), 0)
    return concat_row, row, encoded, indices
    
def make_batch_0(rows, n_col):
    encoded = torch.ones(rows.size())
    for j, row in enumerate(rows):
        _, this_row, enc, _ = make_subset_0(row, n_col)

        rows[j] = this_row
#         print(this_row.size(), enc.size())
        encoded[j] = enc
    return rows, encoded 

def make_condition(input_data, cols):

    encoded = torch.ones(input_data.size())
    encoded[:, cols] = 0
    input_data[:, cols] = 0
    # if len(cols) > 0:
    #     for j in range(input_data.size()[1]):
    #         if j in cols:
    #             input_data[:, j] = 0
    #             encoded[:, j] = 0
    input_data = torch.cat((input_data, encoded), 1)
    return input_data

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
    

    # all_input, all_mask = torch.empty([0]), torch.empty([0])
    # for i in range(num_batches):
    #     input_data = initialize_synthetic_data(BATCH_SIZE, model_kwargs)
    #     a, b = make_batch_0(input_data, 0)
    #     c = torch.cat((a, b), 1)
    #     if train is False:
    #         d = make_condition(input_data, conds)
    #     else:
    #         splits = [0]
    #         dd = []
    #         conds = []
    #         for j in range(model_kwargs['x_dim']+1):

    #             input_tmp = []
    #             if j != model_kwargs['x_dim']:
    #                 splits.append(torch.randint(splits[j], BATCH_SIZE, (1,)).item())
    #                 input_tmp = input_data[splits[j]:splits[j+1], :]
    #             else:
    #                 input_tmp = input_data[splits[j]:BATCH_SIZE, :]

        
    #             if len(conds) != model_kwargs['x_dim'] and len(conds) != 0:
    #                 new_conds = torch.randint(0, model_kwargs['x_dim'] + 1, (model_kwargs['x_dim'],))
    #                 new_conds = [i.item() for i in new_conds]
    #                 while len(set(new_conds)) != len(new_conds):
    #                     new_conds = torch.randint(0, model_kwargs['x_dim'] + 1, (model_kwargs['x_dim'],))
    #                     new_conds = [i.item() for i in new_conds]
    #                 tmp_dd = make_condition(input_tmp, new_conds)
                        
    #             else:
    #                 tmp_dd = make_condition(input_tmp, conds)

    #             dd.append(tmp_dd)
    #             conds.append(j)

    #         d = torch.empty([0])
    #         for k in dd:
    #             d = torch.cat((d, k), 0)
            
    #     c = c.view([1, -1, model_kwargs['x_dim']*2])
    #     d = d.view([1, -1, model_kwargs['x_dim']*2])
    #     all_input = torch.cat((all_input, c), 0)
    #     all_mask = torch.cat((all_mask, d), 0)

    return Batches_X, Batches_C, Batches_conds

def make_synthetic_data_2(num_batches, BATCH_SIZE, conds, model_kwargs):
    all_input, all_mask = torch.empty([0]), torch.empty([0])
    for i in range(num_batches):
        input_data = initialize_synthetic_data(BATCH_SIZE, model_kwargs)
        d = make_condition(input_data, [])
        input_data = input_data.view([1, -1, model_kwargs['x_dim']])
        d = d.view([1, -1, model_kwargs['x_dim']*2])
        all_input = torch.cat((all_input, input_data), 0)
        all_mask = torch.cat((all_mask, d), 0)
    # print(all_input.size(),all_mask.size())
    return all_input, all_mask

def make_synthetic_data_3(num_batches, BATCH_SIZE, conds, model_kwargs):
    all_input, all_mask = torch.empty([0]), torch.empty([0])
    for i in range(num_batches):
        input_data = initialize_synthetic_data(BATCH_SIZE, model_kwargs)
        d = make_condition(input_data, [])
        input_data = input_data.view([1, -1, model_kwargs['x_dim']])
        d = d.view([1, -1, model_kwargs['x_dim']*2])
        d = d[:,:,model_kwargs['x_dim']:]
        all_input = torch.cat((all_input, input_data), 0)
        all_mask = torch.cat((all_mask, d), 0)
    # print(all_input.size(),all_mask.size())
    return all_input, all_mask



