""" A plug and play Spatial Transformer Module in Pytorch """ 
import os 
import numpy as np 
import torch 
import torch.nn as nn 
import torch.nn.functional as F 




class SpatialTransformer(nn.Module):
    """
    Implements a spatial transformer 
    as proposed in the Jaderberg paper. 
    Comprises of 3 parts:
    1. Localization Net
    2. A grid generator 
    3. A roi pooled module.

    The current implementation uses a very small convolutional net with 
    2 convolutional layers and 2 fully connected layers. Backends 
    can be swapped in favor of VGG, ResNets etc. TTMV
    Returns:
    A roi feature map with the same input spatial dimension as the input feature map. 
    """
    def __init__(self, in_channels, spatial_dims, kernel_size,use_dropout=False):
        super(SpatialTransformer, self).__init__()
        self._h, self._w = spatial_dims 
        self._in_ch = in_channels 
        self._ksize = kernel_size
        self.dropout = use_dropout

        # localization net 
        self.conv1 = nn.Conv2d(in_channels, 32, kernel_size=self._ksize, stride=1, padding=1, bias=False) # size : [1x3x32x32]
        self.conv2 = nn.Conv2d(32, 32, kernel_size=self._ksize, stride=1, padding=1, bias=False)
        self.conv3 = nn.Conv2d(32, 32, kernel_size=self._ksize, stride=1, padding=1, bias=False)
        self.conv4 = nn.Conv2d(32, 32, kernel_size=self._ksize, stride=1, padding=1, bias=False)

        self.fc1 = nn.Linear(32*4*4, 1024)
        self.fc2 = nn.Linear(1024, 6)


    def forward(self, x): 
        """
        Forward pass of the STN module. 
        x -> input feature map 
        """
        batch_images = x
        x = F.relu(self.conv1(x.detach()))
        x = F.relu(self.conv2(x))
        x = F.max_pool2d(x, 2)
        x = F.relu(self.conv3(x))
        x = F.max_pool2d(x,2)
        x = F.relu(self.conv3(x))
        x = F.max_pool2d(x, 2)
        # print("Pre view size:{}".format(x.size()))
        x = x.view(-1, 32*4*4)
        if self.dropout:
            x = F.dropout(self.fc1(x), p=0.5)
            x = F.dropout(self.fc2(x), p=0.5)
        else:
            x = self.fc1(x)
            x = self.fc2(x) # params [Nx6]
        
        x = x.view(-1, 2,3) # change it to the 2x3 matrix 
        # print(x.size())
        affine_grid_points = F.affine_grid(x, torch.Size((x.size(0), self._in_ch, self._h, self._w)))
        assert(affine_grid_points.size(0) == batch_images.size(0)), "The batch sizes of the input images must be same as the generated grid."
        rois = F.grid_sample(batch_images, affine_grid_points)
        # print("rois found to be of size:{}".format(rois.size()))
        return rois, affine_grid_points
        

class LocalNetwork(nn.Module):
    def __init__(self, channel, height, width, drop_prob=0.2):
        super(LocalNetwork, self).__init__()
        self.channel = channel
        self.height = height
        self.width = width
        self.drop_prob = drop_prob

        self.fc = nn.Sequential(
            nn.Linear(in_features=self.channel * self.height * self.width,
                      out_features=20),
            nn.Tanh(),
            nn.Dropout(drop_prob),
            nn.Linear(in_features=20, out_features=6),
            nn.Tanh(),
        )
        bias = torch.from_numpy(np.array([1, 0, 0, 0, 1, 0]))

        nn.init.constant_(self.fc[3].weight, 0)
        self.fc[3].bias.data.copy_(bias)

    def forward(self, img):
        '''

        :param img: (b, c, h, w)
        :return: (b, c, h, w)
        '''
        batch_size = img.size(0)

        theta = self.fc(img.view(batch_size, -1)).view(batch_size, 2, 3)

        grid = F.affine_grid(theta, torch.Size((batch_size, self.channel, self.height, self.width)))
        img_transform = F.grid_sample(img, grid)

        return img_transform
        

        

