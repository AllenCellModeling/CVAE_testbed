import torch
import pandas as pd
from Gaussian_CVAE.main_train import str_to_object

def run_synthetic(args, X_train, C_train, Cond_indices_train, X_test, C_test, Cond_indices_test, 
                    n_epochs, loss_fn, model, optimizer, batch_size, gpu_id, model_kwargs): 

    dataframe = {'epoch': [], 'total_train_losses': [], 'total_train_rcl': [], 'total_train_klds': [], 
                'total_test_losses': [], 'total_test_rcl': [], 'total_test_klds': [], 'num_conds': [], 
                'train_rcl': [], 'train_kld': [], 'test_rcl': [], 'test_kld': []}

    dataframe2 = {'epoch': [],'dimension':[], 'test_kld_per_dim': [], 'num_conds': []}


    for i in range(n_epochs):
        print('Training')
        train_loss, train_rcl, train_kld, train_rcl_per_cond, train_kld_per_cond, batch_rcl, batch_kld, batch_num = train(args,
                                                                                         i, loss_fn, X_train,  
                                                                                        C_train, Cond_indices_train, 
                                                                                        batch_size, model, optimizer, 
                                                                                        gpu_id, model_kwargs)
        print('Testing')                                                    
        test_loss, test_rcl, test_kld, test_rcl_per_cond, test_kld_per_cond, test_batch_rcl, test_batch_kld, batch_num_test = test(args,
                                                                                         i, loss_fn, X_test, 
                                                                                         C_test, Cond_indices_test,
                                                                                        batch_size, model, optimizer,
                                                                                         gpu_id, model_kwargs)

        for j in range(len(train_rcl_per_cond)):
            dataframe['epoch'].append(i)
            dataframe['num_conds'].append(j)
            dataframe['total_train_losses'].append(train_loss)
            dataframe['total_train_rcl'].append(train_rcl)
            dataframe['total_train_klds'].append(train_kld)
            dataframe['train_rcl'].append(train_rcl_per_cond[j].item())
            dataframe['train_kld'].append(train_kld_per_cond[j].item())
            dataframe['total_test_losses'].append(test_loss)
            dataframe['total_test_rcl'].append(test_rcl)
            dataframe['total_test_klds'].append(test_kld)
            dataframe['test_rcl'].append(test_rcl_per_cond[j].item())
            dataframe['test_kld'].append(test_kld_per_cond[j].item())


        stats = pd.DataFrame(dataframe)

        # print(batch_kld.size())
        # print(batch_rcl.size())
        # print(test_batch_kld.size())
        for j in range(len(train_rcl_per_cond)):
            # print('outside', batch_num_test)
            # print(j)

            # print(test_batch_kld.size())
            # print(test_batch_rcl.size())
            this_cond_rcl = test_batch_rcl[j::args.model_kwargs['x_dim'] + 1, :]
            this_cond_kld = test_batch_kld[j::args.model_kwargs['x_dim'] + 1, :]
            # print(this_cond_rcl.size(), this_cond_kld.size())
            # print(this_cond_positions)
            summed_rcl = torch.sum(this_cond_rcl, dim = 0)/batch_num_test
            summed_kld = torch.sum(this_cond_kld, dim = 0)/batch_num_test


            for k in range(len(summed_kld)):
                dataframe2['epoch'].append(i)
                dataframe2['dimension'].append(k)
                dataframe2['num_conds'].append(j)
                dataframe2['test_kld_per_dim'].append(summed_kld[ k].item())
        stats_per_dim = pd.DataFrame(dataframe2)

    return stats, stats_per_dim

def train(args, epoch, loss_fn, X_train, C_train, Cond_indices_train, batch_size, model, optimizer, gpu_id, model_kwargs):
    model.train()
    train_loss, rcl_loss, kld_loss = 0, 0, 0
    rcl_per_condition_loss, kld_per_condition_loss = torch.zeros(X_train.size()[-1] + 1), torch.zeros(X_train.size()[-1]  + 1)
    batch_cond, batch_rcl, batch_kld = torch.empty([0]), torch.empty([0]), torch.empty([0])
    batch_length = 0

    for j, i in enumerate(X_train):
        optimizer.zero_grad()
        c, d, cond_labels = X_train[j], C_train[j], Cond_indices_train[j]

        recon_batch, mu, log_var = model(c.cuda(gpu_id), d.cuda(gpu_id))

        loss_fn = str_to_object(args.loss_fn)
        loss, rcl, kld, rcl_per_element, kld_per_element = loss_fn(c.cuda(gpu_id), recon_batch.cuda(gpu_id), mu, log_var)       

        loss.backward()
        train_loss += loss.item()
        rcl_loss += rcl.item()
        kld_loss += kld.item()
        
        # print('train', len(X_train))

        for jj, ii in enumerate(torch.unique(cond_labels)):
            # print(batch_kld.size())

            this_cond_positions = cond_labels == ii
            # print(kld_per_element[this_cond_positions].size())
            # print(torch.sum(kld_per_element[this_cond_positions], dim = 0).size())
            # print(X_train.size(), cond_labels.size(), torch.unique(cond_labels))
            if len(torch.unique(cond_labels)) == c.size()[-1] + 1:
                batch_rcl = torch.cat([batch_rcl.cuda(gpu_id), torch.sum(rcl_per_element[this_cond_positions], dim = 0).view(1, -1)], 0)
                batch_kld = torch.cat([batch_kld.cuda(gpu_id), torch.sum(kld_per_element[this_cond_positions], dim = 0).view(1, -1)], 0)
                batch_length += 1

            this_cond_rcl = torch.sum(rcl_per_element[this_cond_positions])
            this_cond_kld = torch.sum(kld_per_element[this_cond_positions])

            rcl_per_condition_loss[jj] += this_cond_rcl.item()
            kld_per_condition_loss[jj] += this_cond_kld.item()
        optimizer.step()   
    # print('END', batch_kld.size())
    num_batches = len(X_train)    
    print('====> Epoch: {} Train loss: {:.4f}'.format(epoch, train_loss/num_batches))
    print('====> Train RCL loss: {:.4f}'.format(rcl_loss/num_batches))
    print('====> Train KLD loss: {:.4f}'.format(kld_loss/num_batches))

    batch_rcl, batch_kld = batch_rcl, batch_kld
    
    return train_loss/num_batches, rcl_loss/num_batches, kld_loss/num_batches, rcl_per_condition_loss/num_batches, kld_per_condition_loss/num_batches, batch_rcl, batch_kld, batch_length

def test(args, epoch, loss_fn, X_test, C_test,Cond_indices_test, batch_size, model, optimizer, gpu_id, model_kwargs):

    model.eval()
    test_loss, rcl_loss, kld_loss = 0, 0, 0
    rcl_per_condition_loss, kld_per_condition_loss = torch.zeros(X_test.size()[-1]  + 1), torch.zeros(X_test.size()[-1]  + 1)
    batch_cond, batch_rcl, batch_kld = torch.empty([0]), torch.empty([0]), torch.empty([0])
    batch_length = 0

    with torch.no_grad():
        for j, i in enumerate(X_test):
            optimizer.zero_grad()
            c, d, cond_labels = X_test[j], C_test[j], Cond_indices_test[j] 

            recon_batch, mu, log_var = model(c.cuda(gpu_id), d.cuda(gpu_id))
            loss_fn = str_to_object(args.loss_fn)
            loss, rcl, kld, rcl_per_element, kld_per_element = loss_fn(c.cuda(gpu_id), recon_batch.cuda(gpu_id), mu, log_var)         
            test_loss += loss.item()
            rcl_loss += rcl.item()
            kld_loss += kld.item()
            # print(X_test.size(), cond_labels.size(), torch.unique(cond_labels))
            for jj, ii in enumerate(torch.unique(cond_labels)):

                this_cond_positions = cond_labels == ii
                if len(torch.unique(cond_labels)) == c.size()[-1] + 1:
                    batch_rcl = torch.cat([batch_rcl.cuda(gpu_id), torch.sum(rcl_per_element[this_cond_positions], dim = 0).view(1, -1)], 0)
                    batch_kld = torch.cat([batch_kld.cuda(gpu_id), torch.sum(kld_per_element[this_cond_positions], dim = 0).view(1, -1)], 0)
                    batch_length += 1
    
                this_cond_rcl = torch.sum(rcl_per_element[this_cond_positions])
                this_cond_kld = torch.sum(kld_per_element[this_cond_positions])
                rcl_per_condition_loss[jj] += this_cond_rcl.item()
                kld_per_condition_loss[jj] += this_cond_kld.item()

    num_batches = len(X_test)
    # print(batch_kld.size(), 'test')
    # print('loss function', args.loss_fn)
    print('====> Epoch: {} Test losses: {:.4f}'.format(epoch, test_loss/num_batches))
    print('====> RCL loss: {:.4f}'.format( rcl_loss/num_batches))
    print('====> KLD loss: {:.4f}'.format( kld_loss/num_batches))


    batch_rcl, batch_kld = batch_rcl, batch_kld

    return test_loss/num_batches, rcl_loss/num_batches, kld_loss/num_batches,rcl_per_condition_loss/num_batches, kld_per_condition_loss/num_batches, batch_rcl, batch_kld, batch_length

