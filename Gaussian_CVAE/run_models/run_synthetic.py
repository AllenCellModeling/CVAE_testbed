import torch
import pandas as pd
from datasets.dataloader import make_synthetic_data
from main_train import str_to_object

def run_synthetic(args, X_train, C_train, Cond_indices_train, X_test, C_test, Cond_indices_test, 
                    n_epochs, loss_fn, model, optimizer, batch_size, gpu_id, model_kwargs): 

    dataframe = {'epoch': [], 'total_train_losses': [], 'total_train_rcl': [], 'total_train_klds': [], 
                'total_test_losses': [], 'total_test_rcl': [], 'total_test_klds': [], 'num_conds': [], 
                'train_rcl': [], 'train_kld': [], 'test_rcl': [], 'test_kld': []}

    dataframe2 = {'epoch': [],'dimension':[],  'train_kld_per_dim': [],  'test_kld_per_dim': [], 'num_conds': []}


    for i in range(n_epochs):
        print('Training')
        train_loss, train_rcl, train_kld, train_rcl_per_cond, train_kld_per_cond, rcls, klds = train(args,
                                                                                         i, loss_fn, X_train,  
                                                                                        C_train, Cond_indices_train, 
                                                                                        batch_size, model, optimizer, 
                                                                                        gpu_id, model_kwargs)
        print('Testing')                                                    
        test_loss, test_rcl, test_kld, test_rcl_per_cond, test_kld_per_cond, test_rcls, test_klds = test(args,
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
        print(len(train_rcl_per_cond))
        for j in range(len(train_rcl_per_cond)):
            print('outside here')
            this_cond_rcl = rcls[j]
            this_cond_kld = klds[j]
            this_cond_rcl_test = test_rcls[j]
            this_cond_kld_test = test_klds[j]
            print(this_cond_rcl.size())
            print(this_cond_kld.size())
            print(this_cond_rcl_test.size())
            print(this_cond_kld_test.size())
            print('LOOK HERE')
            # print(this_cond_rcl)
            # print(this_cond_kld)
            for k in range(len(this_cond_kld)):
                dataframe2['epoch'].append(i)
                dataframe2['dimension'].append(k)
                dataframe2['num_conds'].append(j)
                dataframe2['train_kld_per_dim'].append(this_cond_kld[k].item())
                dataframe2['test_kld_per_dim'].append(this_cond_kld_test[ k].item())
        stats_per_dim = pd.DataFrame(dataframe2)

    return stats, stats_per_dim

def train(args, epoch, loss_fn, X_train, C_train, Cond_indices_train, batch_size, model, optimizer, gpu_id, model_kwargs):
    model.train()
    train_loss, rcl_loss, kld_loss = 0, 0, 0
    rcl_per_condition_loss, kld_per_condition_loss = torch.zeros(model_kwargs['x_dim'] + 1), torch.zeros(model_kwargs['x_dim'] + 1)
    conds_rcl_1, conds_kld_1, conds_rcl_2, conds_kld_2, conds_rcl_3, conds_kld_3 = torch.empty([0]), torch.empty([0]), torch.empty([0]), torch.empty([0]), torch.empty([0]), torch.empty([0])
    batch_cond = torch.empty([0])
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
        
        for jj, ii in enumerate(torch.unique(cond_labels)):
            # print(ii, torch.unique(cond_labels), rcl_per_element.size())
            this_cond_positions = cond_labels == ii

            # batch_cond =  torch.cat([batch_cond, this_cond_positions], 0)
            # batch_rcl = torch.cat([batch_rcl, rcl_per_element], 0)
            # batch_kld = torch.cat([batch_kld, kld_per_element], 0)

            if ii == 0:
                # print(torch.sum(kld_per_element[this_cond_positions], dim = 0).view(1, -1).size())
                conds_rcl_1 = torch.cat([conds_rcl_1, torch.sum(rcl_per_element[this_cond_positions], dim = 0).view(1, -1).cpu()], 0)
                conds_kld_1 = torch.cat([conds_kld_1, torch.sum(kld_per_element[this_cond_positions], dim = 0).view(1,-1).cpu()], 0)
                # print(conds_rcl_1.size())
                # print(ii, conds_kld_1.size())
            elif ii == 1:
                conds_rcl_2 = torch.cat([conds_rcl_2, torch.sum(rcl_per_element[this_cond_positions], dim = 0).view(1, -1).cpu()], 0)
                conds_kld_2 = torch.cat([conds_kld_2, torch.sum(kld_per_element[this_cond_positions], dim = 0).view(1,-1).cpu()], 0)
                # print(ii, conds_kld_2.size())
            elif ii == 2:
                conds_rcl_3 = torch.cat([conds_rcl_3, torch.sum(rcl_per_element[this_cond_positions], dim = 0).view(1, -1).cpu()], 0)
                conds_kld_3 = torch.cat([conds_kld_3, torch.sum(kld_per_element[this_cond_positions], dim = 0).view(1, -1).cpu()], 0)
                # print(ii, conds_kld_3.size())
            this_cond_rcl = torch.sum(rcl_per_element[this_cond_positions])
            this_cond_kld = torch.sum(kld_per_element[this_cond_positions])
            rcl_per_condition_loss[jj] += this_cond_rcl.item()
            kld_per_condition_loss[jj] += this_cond_kld.item()


        optimizer.step()   
    num_batches = len(X_train)    
    print('====> Epoch: {} Train loss: {:.4f}'.format(epoch, train_loss/num_batches))
    print('====> Train RCL loss: {:.4f}'.format(rcl_loss/num_batches))
    print('====> Train KLD loss: {:.4f}'.format(kld_loss/num_batches))
    
    
    # print(conds_rcl_1.size())
    conds_rcl_1 = torch.sum(conds_rcl_1, dim = 0)/num_batches
    # print(conds_rcl_1.size())
    # print(conds_kld_1.size())
    conds_rcl_2 = torch.sum(conds_rcl_2, dim = 0)/(num_batches)
    conds_rcl_3 = torch.sum(conds_rcl_3, dim = 0)/num_batches
    conds_kld_1 = torch.sum(conds_kld_1, dim = 0)/num_batches
    # print(conds_kld_1.size())
    conds_kld_2 = torch.sum(conds_kld_2, dim = 0)/(num_batches)
    conds_kld_3 = torch.sum(conds_kld_3, dim = 0)/num_batches

    # print(conds_rcl_1)
    rcls, klds = [], []
    rcls.append(conds_rcl_1)
    rcls.append(conds_rcl_2)
    rcls.append(conds_rcl_3)
    klds.append(conds_kld_1)
    klds.append(conds_kld_2)
    klds.append(conds_kld_3)

    return train_loss/num_batches, rcl_loss/num_batches, kld_loss/num_batches, rcl_per_condition_loss/num_batches, kld_per_condition_loss/num_batches, rcls, klds

def test(args, epoch, loss_fn, X_test, C_test,Cond_indices_test, batch_size, model, optimizer, gpu_id, model_kwargs):

    model.eval()
    test_loss, rcl_loss, kld_loss = 0, 0, 0
    rcl_per_condition_loss, kld_per_condition_loss = torch.zeros(model_kwargs['x_dim'] + 1), torch.zeros(model_kwargs['x_dim'] + 1)
    conds_rcl_1, conds_kld_1, conds_rcl_2, conds_kld_2, conds_rcl_3, conds_kld_3 = torch.empty([0]), torch.empty([0]), torch.empty([0]), torch.empty([0]), torch.empty([0]), torch.empty([0])
    print('inside_test')
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
            for jj, ii in enumerate(torch.unique(cond_labels)):

                this_cond_positions = cond_labels == ii
    
                if ii == 0:
                    # print(torch.sum(kld_per_element[this_cond_positions], dim = 0).view(1, -1).size())
                    conds_rcl_1 = torch.cat([conds_rcl_1, torch.sum(rcl_per_element[this_cond_positions], dim = 0).view(1, -1).cpu()], 0)
                    conds_kld_1 = torch.cat([conds_kld_1, torch.sum(kld_per_element[this_cond_positions], dim = 0).view(1,-1).cpu()], 0)
                    # print(conds_rcl_1.size())
                    # print(conds_kld_1.size())
                elif ii == 1:
                    conds_rcl_2 = torch.cat([conds_rcl_2, torch.sum(rcl_per_element[this_cond_positions], dim = 0).view(1, -1).cpu()], 0)
                    conds_kld_2 = torch.cat([conds_kld_2, torch.sum(kld_per_element[this_cond_positions], dim = 0).view(1,-1).cpu()], 0)
                elif ii == 2:
                    conds_rcl_3 = torch.cat([conds_rcl_3, torch.sum(rcl_per_element[this_cond_positions], dim = 0).view(1, -1).cpu()], 0)
                    conds_kld_3 = torch.cat([conds_kld_3, torch.sum(kld_per_element[this_cond_positions], dim = 0).view(1, -1).cpu()], 0)
    
                this_cond_rcl = torch.sum(rcl_per_element[this_cond_positions])
                this_cond_kld = torch.sum(kld_per_element[this_cond_positions])
                rcl_per_condition_loss[jj] += this_cond_rcl.item()
                kld_per_condition_loss[jj] += this_cond_kld.item()
    num_batches = len(X_test)
    # print('loss function', args.loss_fn)
    print('====> Epoch: {} Test losses: {:.4f}'.format(epoch, test_loss/num_batches))
    print('====> RCL loss: {:.4f}'.format( rcl_loss/num_batches))
    print('====> KLD loss: {:.4f}'.format( kld_loss/num_batches))
    conds_rcl_1 = torch.sum(conds_rcl_1, dim = 0)/num_batches
    conds_rcl_2 = torch.sum(conds_rcl_2, dim = 0)/(num_batches)
    conds_rcl_3 = torch.sum(conds_rcl_3, dim = 0)/num_batches
    conds_kld_1 = torch.sum(conds_kld_1, dim = 0)/num_batches
    conds_kld_2 = torch.sum(conds_kld_2, dim = 0)/(num_batches)
    conds_kld_3 = torch.sum(conds_kld_3, dim = 0)/num_batches
    print('Cond KLD 1', cond_kld_1)

    # print(conds_rcl_1)
    rcls, klds = [], []
    rcls.append(conds_rcl_1)
    rcls.append(conds_rcl_2)
    rcls.append(conds_rcl_3)
    klds.append(conds_kld_1)
    klds.append(conds_kld_2)
    klds.append(conds_kld_3)

    return test_loss/num_batches, rcl_loss/num_batches, kld_loss/num_batches,rcl_per_condition_loss/num_batches, kld_per_condition_loss/num_batches, rcls, klds

