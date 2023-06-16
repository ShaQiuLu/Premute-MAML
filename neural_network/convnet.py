import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import OrderedDict

class ConvNet(nn.Module):

    def __init__(self, in_channels, hid_dim=64, z_dim=64):
        super().__init__()
        self.num_layers = 4
        self.is_training = True
        # input layer, add_module:for register
        self.add_module('{0}_{1}'.format(0,0), nn.Conv2d(in_channels, hid_dim, 3, padding=1))
        self.add_module('{0}_{1}'.format(0,1), nn.BatchNorm2d(hid_dim))
        # hidden layer
        for i in [1, 2]:
            self.add_module('{0}_{1}'.format(i,0), nn.Conv2d(hid_dim, hid_dim, 3, padding=1))   
            self.add_module('{0}_{1}'.format(i,1), nn.BatchNorm2d(hid_dim))     
        # last layer
        self.add_module('{0}_{1}'.format(3,0), nn.Conv2d(hid_dim, z_dim, 3, padding=1))   
        self.add_module('{0}_{1}'.format(3,1), nn.BatchNorm2d(z_dim))
        self.add_module('fc', nn.Linear(64, 5))

        # for m in self.modules():
        #     if isinstance(m, nn.Conv2d):
        #         nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='leaky_relu')
        #     elif isinstance(m, nn.BatchNorm2d):
        #         nn.init.constant_(m.weight, 1)
        #         nn.init.constant_(m.bias, 0)
    
    def forward(self, x, params = None, embedding = False):
        if params is None:
            params = OrderedDict(self.named_parameters())
            
        output = x
        for i in range(self.num_layers):
            output = F.conv2d(output, params['{0}_{1}.weight'.format(i,0)], bias=params['{0}_{1}.bias'.format(i,0)], padding=1)
            output = F.batch_norm(output, weight=params['{0}_{1}.weight'.format(i,1)], bias=params['{0}_{1}.bias'.format(i,1)],
                                  running_mean=self._modules['{0}_{1}'.format(i,1)].running_mean,
                                  running_var=self._modules['{0}_{1}'.format(i,1)].running_var, training = self.is_training)
            output = F.relu(output)
            output = F.max_pool2d(output, 2)
        # print(output.shape)

        output = F.avg_pool2d(output, 5)     # AveragePool Here
        # print(output.shape)
        output = output.view(x.size(0), -1)
        print(output.shape)
        
        if embedding:
            return output
        else:
            # Apply Linear Layer
            logits = F.linear(output, weight=params['fc.weight'], bias=params['fc.bias'])
            return logits

if __name__ == '__main__':
    model = ConvNet(3)
    x = torch.rand(3, 3, 96, 96)
    pred = model(x)
    # print(pred)
    # params = OrderedDict(model.named_parameters())
    # for i in params:
    #     print(i)
    # param = model.parameters()
    # for i in param:
    #     print(i)
    # model.fc = nn.Linear(64, 5)
    # fcone = nn.Linear(64, 1)
    # model.fc.weight.data = fcone.weight.data.repeat(5, 1)
    # model.fc.bias.data = fcone.bias.data.repeat(5)
    # params = OrderedDict(model.named_parameters())
    # print(params)
