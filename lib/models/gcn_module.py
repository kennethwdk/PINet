import torch
from torch import nn

import math

class ResGraphBlock(nn.Module):
    def __init__(self, input_dim, output_dim, hid_dim):
        super(ResGraphBlock, self).__init__()
        self.gconv1 = BasicGraphBlock(input_dim, hid_dim)
        self.gconv2 = BasicGraphBlock(hid_dim, output_dim)

    def forward(self, x, adj=None):
        residual = x
        out = self.gconv1(x, adj)
        out = self.gconv2(out, adj)
        return residual + out

class BasicGraphBlock(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(BasicGraphBlock, self).__init__()
        self.conv = GraphConv(input_dim, output_dim)
        self.bn = nn.BatchNorm1d(output_dim)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x, adj=None):
        x = self.conv(x, adj).transpose(1, 2)
        x = self.bn(x).transpose(1, 2)
        x = self.relu(x)
        return x

class GraphConv(nn.Module):
    def __init__(self, input_dim, output_dim, bias=True):
        super(GraphConv, self).__init__()
        self.in_features = input_dim
        self.out_features = output_dim
        self.weight = nn.Parameter(torch.zeros(input_dim, output_dim), requires_grad=True)
        nn.init.xavier_uniform_(self.weight.data, gain=1.414)
        if bias:
            self.bias = nn.Parameter(torch.zeros(output_dim), requires_grad=True)
            stdv = 1. / math.sqrt(self.weight.size(1))
            self.bias.data.uniform_(-stdv, stdv)
        else:
            self.register_parameter('bias', None)

    def forward(self, x, adj):
        x = torch.matmul(x, self.weight)
        x = torch.matmul(adj, x)
        if self.bias is not None:
            x += self.bias.view(1, 1, -1)
        return x

class EdgeConv(nn.Module):
    """
    Edge convolution layer (with activation, batch normalization) for dense data type
    """
    def __init__(self, in_channels, out_channels, bias=True):
        super(EdgeConv, self).__init__()
        self.nn = nn.Sequential(
            nn.Conv2d(in_channels*2, out_channels, 1, bias=bias),
            nn.ReLU(True),
            nn.BatchNorm2d(out_channels, affine=True)
        )

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm2d) or isinstance(m, nn.InstanceNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def forward(self, x, k=None):
        with torch.no_grad():
            edge_index = knn(x.squeeze(3), k)
        x_i = batched_index_select(x, edge_index[1])
        x_j = batched_index_select(x, edge_index[0])
        max_value, _ = torch.max(self.nn(torch.cat([x_i, x_j - x_i], dim=1)), -1, keepdim=True)
        return max_value

def knn(x, k):
    '''
    Args:
        x: torch.Size([batch_size, num_dims, num_vertices])
        k: neighborhood
    Returns:
    '''
    inner = -2 * torch.matmul(x.transpose(2, 1), x)
    xx = torch.sum(x ** 2, dim=1, keepdim=True)
    pairwise_distance = -xx - inner - xx.transpose(2, 1)
    idx = pairwise_distance.topk(k=k, dim=-1)[1]  # (batch_size, num_points, k)
    idx_base = torch.arange(idx.size(1)).unsqueeze(1).expand(-1, k)
    idx_base = idx_base.unsqueeze(0).expand(idx.size(0), -1, -1).to(x.device)
    return [idx, idx_base]

def batched_index_select(inputs, index):
    """
    :param inputs: torch.Size([batch_size, num_dims, num_vertices, 1])
    :param index: torch.Size([batch_size, num_vertices, k])
    :return: torch.Size([batch_size, num_dims, num_vertices, k])
    """
    batch_size, num_dims, num_vertices, _ = inputs.shape
    k = index.shape[2]
    idx = torch.arange(0, batch_size) * num_vertices
    idx = idx.view(batch_size, -1)

    inputs = inputs.transpose(2, 1).contiguous().view(-1, num_dims)
    index = index.view(batch_size, -1) + idx.type(index.dtype).to(inputs.device)
    index = index.view(-1)

    return torch.index_select(inputs, 0, index).view(batch_size, -1, num_dims).transpose(2, 1).view(batch_size, num_dims, -1, k)