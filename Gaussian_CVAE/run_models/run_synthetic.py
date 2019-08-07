import torch
import pandas as pd
from datasets.dataloader import make_synthetic_data
from main_train import str_to_object

def run_synthetic(args, all_input_train, all_mask_train, all_input_test, all_mask_test, 
                    n_epochs, loss_fn, model, optimizer, batch_size, gpu_id, model_kwargs): 
    dataframe = {'epoch': [], 'train_losses': [], 'train_rcl': [], 'train_klds': [], 
                'test_losses': [], 'test_rcl': [], 'test_klds': [], 'num_conds': []}

    conds = []

    for j in range(model_kwargs['x_dim'] + 1):
        print('Number of conditions', model_kwargs['x_dim']-len(conds))
        for i in range(n_epochs):
            print('Training')
            train_loss, train_rcl, train_kld = train(args, i, loss_fn, all_input_train,  all_mask_train, 
                                                    batch_size, model, optimizer, gpu_id,  conds, model_kwargs)
            print('Testing')
            test_loss, test_rcl, test_kld = test(args, i, loss_fn, all_input_test, all_mask_test, 
                                                    batch_size, model, optimizer, gpu_id, conds, model_kwargs)
            # print(test_loss)
            dataframe['epoch'].append(i)
            dataframe['train_losses'].append(train_loss)
            dataframe['train_rcl'].append(train_rcl)
            dataframe['train_klds'].append(train_kld)
            dataframe['test_losses'].append(test_loss)
            dataframe['test_rcl'].append(test_rcl)
            dataframe['test_klds'].append(test_kld)
            dataframe['num_conds'].append(model_kwargs['x_dim'] - len(conds))
        conds.append(j)
    stats = pd.DataFrame(dataframe)
    return stats

def train(args, epoch, loss_fn, all_input_train, all_mask_train, batch_size, model, optimizer, gpu_id, conds, model_kwargs):
    model.train()
    train_loss, rcl_loss, kld_loss = 0, 0, 0
    for j, i in enumerate(all_input_train):
        optimizer.zero_grad()
        c, d = all_input_train[j], all_mask_train[j]

        if args.dataloader != 'datasets.dataloader.make_synthetic_data_3': 
            # print(args.dataloader)
            if len(conds) > 0:
                tmp1, tmp2 = torch.split(d, 2, dim=1)
                for kk in conds:
                    tmp1[:, kk], tmp2[:, kk] = 0, 0
                d = torch.cat((tmp1, tmp2), 1)
        else:
            for kk in conds:
                d[:, kk] =0

        recon_batch, mu, log_var = model(c.cuda(gpu_id), d.cuda(gpu_id))
        # recon_batch = recon_batch.view(-1, model_kwargs['x_dim']*2)
        # print(c.size(), recon_batch.size())
        loss_fn = str_to_object(args.loss_fn)
        loss, rcl, kld = loss_fn(c.cuda(gpu_id), recon_batch.cuda(gpu_id), mu, log_var)       
        loss.backward()
        train_loss += loss.item()
        rcl_loss += rcl.item()
        kld_loss += kld.item()
        optimizer.step()   
    num_batches = len(all_input_train)    
    print('====> Epoch: {} Train loss: {:.4f}'.format(epoch, train_loss/num_batches))
    print('====> Train RCL loss: {:.4f}'.format(rcl_loss/num_batches))
    print('====> Train KLD loss: {:.4f}'.format(kld_loss/num_batches))
    return train_loss/num_batches, rcl_loss/num_batches, kld_loss/num_batches

def test(args, epoch, loss_fn, all_input_test, all_mask_test, batch_size, model, optimizer, gpu_id, conds, model_kwargs):

    model.eval()
    test_loss, rcl_loss, kld_loss = 0, 0, 0
    with torch.no_grad():
        for j, i in enumerate(all_input_test):
            optimizer.zero_grad()
            c, d = all_input_test[j], all_mask_test[j]  
            if args.dataloader != 'datasets.dataloader.make_synthetic_data_3': 
                # print(args.dataloader)
                if len(conds) > 0:
                    tmp1, tmp2 = torch.split(d, 2, dim=1)
                    for kk in conds:
                        tmp1[:, kk], tmp2[:, kk] = 0, 0
                    d = torch.cat((tmp1, tmp2), 1)
            else:
                for kk in conds:
                    d[:, kk] =0

            recon_batch, mu, log_var = model(c.cuda(gpu_id), d.cuda(gpu_id))
            loss_fn = str_to_object(args.loss_fn)
            loss, rcl, kld = loss_fn(c.cuda(gpu_id), recon_batch.cuda(gpu_id), mu, log_var)         
            test_loss += loss.item()
            rcl_loss += rcl.item()
            kld_loss += kld.item()
    num_batches = len(all_input_test)
    # print('loss function', args.loss_fn)
    print('====> Epoch: {} Test losses: {:.4f}'.format(epoch, test_loss/num_batches))
    print('====> RCL loss: {:.4f}'.format( rcl_loss/num_batches))
    print('====> KLD loss: {:.4f}'.format( kld_loss/num_batches))
    return test_loss/num_batches, rcl_loss/num_batches, kld_loss/num_batches

