import torch
from torch import nn
import torch.nn.functional as F
from collections import OrderedDict


def conv3x3(in_channel, out_channel, stride=1):
    return nn.Conv2d(in_channel, out_channel, kernel_size=3, stride=1, padding=1, bias=False)


class BasicBlock(nn.Module):   # 50层以内
    def __init__(self, in_channel, out_channel):   # 64
        super(BasicBlock, self).__init__()

        self.conv1 = conv3x3(in_channel, out_channel)
        self.bn1 = nn.BatchNorm2d(out_channel)
        self.relu = nn.LeakyReLU(0.1)
        self.conv2 = conv3x3(out_channel, out_channel)
        self.bn2 = nn.BatchNorm2d(out_channel)
        self.conv3 = conv3x3(out_channel, out_channel)
        self.bn3 = nn.BatchNorm2d(out_channel)

        self.downsample = nn.Conv2d(in_channel, out_channel,
                      kernel_size=1, stride=1, bias=False)
        self.bn = nn.BatchNorm2d(out_channel)

        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.dropout = nn.Dropout(0.5)

    def forward(self, x, params, base, modules, drop=False):
        residual = x

        x = F.conv2d(x, weight=params[base + '.conv1.weight'], stride=(1, 1), padding=(1, 1))
        x = F.batch_norm(x, weight=params[base + '.bn1.weight'], bias=params[base + '.bn1.bias'],
                           running_mean=modules['bn1'].running_mean,
                           running_var=modules['bn1'].running_var, training = True)
        x = self.relu(x)

        x = F.conv2d(x, weight=params[base + '.conv2.weight'], stride=(1, 1), padding=(1, 1))
        x = F.batch_norm(x, weight=params[base + '.bn2.weight'], bias=params[base + '.bn2.bias'],
                         running_mean=modules['bn2'].running_mean,
                         running_var=modules['bn2'].running_var, training=True)
        x = self.relu(x)


        x = F.conv2d(x, weight=params[base + '.conv3.weight'], stride=(1, 1), padding=(1, 1))
        x = F.batch_norm(x, weight=params[base + '.bn3.weight'], bias=params[base + '.bn3.bias'],
                         running_mean=modules['bn3'].running_mean,
                         running_var=modules['bn3'].running_var, training=True)
        if drop == True:
            x = self.dropout(x)

        residual = F.conv2d(residual, weight=params[base + '.downsample.weight'], stride=(1, 1))
        residual = F.batch_norm(residual, weight=params[base + '.bn.weight'], bias=params[base + '.bn.bias'],
                         running_mean=modules['bn'].running_mean,
                         running_var=modules['bn'].running_var, training=True)
        x += residual
        x = self.relu(x)
        x = self.maxpool(x)

        return x


class ResNet(nn.Module):
    def __init__(self, in_channel):    # 3 10
        super(ResNet, self).__init__()

        self.block = BasicBlock(in_channel, 64)
        self.block2 = BasicBlock(64, 100)
        self.block3 = BasicBlock(100, 230)
        self.block4 = BasicBlock(230, 460)

        self.flatten = nn.Flatten()
        # self.fc = nn.Linear(460*6*6, 5)
        self.dropout = nn.Dropout(0.5)

    def forward(self, x, params=None, embedding = False):
        if params is None:
            params = OrderedDict(self.named_parameters())

        x = self.block(x, params, 'block', self._modules['block']._modules)
        x = self.block2(x, params, 'block2', self._modules['block2']._modules)
        x = self.block3(x, params, 'block3', self._modules['block3']._modules, True)
        x = self.block4(x, params, 'block4', self._modules['block4']._modules, True)
        # print(x.shape)

        x = x.view(x.size(0), -1)

        if embedding:
            return x
        else:
            # Apply Linear Layer
            logits = F.linear(x, weight=params['fc.weight'], bias=params['fc.bias'])
            logits = self.dropout(logits)

            return logits



if __name__ == '__main__':
    fack_img = torch.randint(0, 255, [25, 3, 84, 84]).type(torch.FloatTensor)
    # y = torch.tensor([1,1,1,1,1,0,0,0,0,0,2,2,2,2,2,3,3,3,3,3,4,4,4,4,4])
    resnet = ResNet(3)
    # print(OrderedDict(resnet.named_parameters()))
    pred = resnet(fack_img)




