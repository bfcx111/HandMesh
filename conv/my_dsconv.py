# Copyright (c) Xingyu Chen. All Rights Reserved.

"""
 * @file dsconv.py
 * @author chenxingyu (chenxy.sean@gmail.com)
 * @brief Depth-separable spiral convolution
 * @version 0.1
 * @date 2022-04-28
 *
 * @copyright Copyright (c) 2022 chenxingyu
 *
"""

import torch
import torch.nn as nn
import numpy as np


class DSConv(nn.Module):
    def __init__(self, in_channels, out_channels, indices, dim=1):
        super(DSConv, self).__init__()
        self.dim = dim
        self.indices = indices
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.seq_length = indices.size(1)
        self.spatial_layer = nn.Conv2d(self.in_channels, self.in_channels, int(np.sqrt(self.seq_length)), 1, 0, groups=self.in_channels, bias=False)
        self.channel_layer = nn.Linear(self.in_channels, self.out_channels, bias=False)
        torch.nn.init.xavier_uniform_(self.channel_layer.weight)

    def reset_parameters(self):
        torch.nn.init.xavier_uniform_(self.spatial_layer.weight)
        torch.nn.init.xavier_uniform_(self.channel_layer.weight)

    def forward(self, x):
        n_nodes, _ = self.indices.size()
        bs = x.size(0)
        x = torch.index_select(x, self.dim, self.indices.to(x.device).view(-1))
        x = x.view(bs * n_nodes, self.seq_length, -1).transpose(1, 2)
        x = x.view(x.size(0), x.size(1), int(np.sqrt(self.seq_length)), int(np.sqrt(self.seq_length)))
        # breakpoint()
        # pred = self.spatial_layer(x).reshape(bs, n_nodes, -1)
        wt = self.spatial_layer.weight
        wt = wt.reshape(-1,3,3)
        # yyy = x*wt[None]
        yyy = torch.mul(x,wt[None])
        bbb = yyy.sum(axis=[2, 3], keepdim=False)
        x = bbb.view(bs, n_nodes, -1)
        # test4.append(x)
        # self.spatial_layer.weight
        # x = self.spatial_layer(x)
        # print('xshape:{},  bs:{},  n_nodes:{}'.format(x.shape,bs,n_nodes))
        # x = x.view(bs, n_nodes, -1)
        # x = x[:,:,0,0].unsqueeze(0)
        # x = torch.zeros(1,x.shape[0],x.shape[1]).to('cuda')
        # print('xshape:{},  bs:{},  n_nodes:{}'.format(self.spatial_layer(x).shape,bs,n_nodes))
        # x = self.spatial_layer(x).reshape(bs, n_nodes, -1)
        # x = self.spatial_layer(x).view(bs, n_nodes, -1)
        x = self.channel_layer(x)

        return x

    def __repr__(self):
        return '{}({}, {}, seq_length={})'.format(self.__class__.__name__,
                                                  self.in_channels,
                                                  self.out_channels,
                                                  self.seq_length)