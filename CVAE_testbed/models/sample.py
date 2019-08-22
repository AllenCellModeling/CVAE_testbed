import torch
import sys
sys.path.insert(0, '.')
from CVAE_testbed.models.CVAE_first import idx2onehot

class Sample():
    def __init__(self, x, y, BATCH_SIZE, device):
        """
        Returns a reformatted x and y 
        """
        x = x.repeat(1, 3, 1, 1)      
        self.x = x
        self.y = y
        self.batch_size = BATCH_SIZE
        self.device = device

    def generate_single(self, j):
        return self.color_and_digit(j)

    def generate_x_y(self):
        for k, j in enumerate(range(self.batch_size)):
            #rnd = torch.randint(0, 4, (1,1)).item() 
#             print(tmp_x.size(), tmp_y.size())
            if k ==0:
                tmp_x, tmp_y = self.generate_single(j)

            if k > 0:
                tmp_xt, tmp_yt = self.generate_single(j)

                tmp_x = torch.cat((tmp_x, tmp_xt), dim = 0)
                tmp_y = torch.cat((tmp_y, tmp_yt), dim = 0)

        tmp_x = tmp_x.view(-1, 28 * 28*3)
        tmp_x = tmp_x.to(self.device)
        return tmp_x, tmp_y
    
    def assign_colors(self, j):
        colors = []
        color = torch.randint(1, 4, (1,1)).item()
        other_indices = []
        color_index = []
        for a in [1,2,3]:
            if color != a:
                other_indices.append(a)
            else:
                color_index = a
        self.x[j, other_indices[0]-1, :, :].fill_(0)
        self.x[j, other_indices[1]-1, :, :].fill_(0)
        colors.append(color-1) 

        return self.x[j, :,:,:].view(-1,3,28,28), colors
    
    def make_y2(self, colors):
        y2 = torch.LongTensor(colors)
        y2 = idx2onehot(y2.view(-1, 1), n=3)
        y2 = y2.to(self.device)
        return y2

    def make_y(self, y):
        y = idx2onehot(y.view(-1, 1), n = 10)
        y = y.to(self.device)
        return y 

    def color_and_digit(self, j):
        
        this_x, colors = self.assign_colors(j)            
        y2 = self.make_y2(colors)    
        this_y = self.make_y(self.y[j])   
        
        this_y = torch.cat((this_y, torch.FloatTensor([0]).view(-1,1).cuda()), dim = 1)
        
        this_y = torch.cat((this_y, y2), dim=1)
        this_y = torch.cat((this_y, torch.FloatTensor([0]).view(-1,1).cuda()), dim = 1)
#         print(this_y.size(),torch.FloatTensor([0, 0]).view(-1,2).size() )
        
        return this_x, this_y
