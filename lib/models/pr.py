import torch
from torch import nn

from .utils import int_sample, float_sample
from .gcn_module import BasicGraphBlock, ResGraphBlock

from dataset import VIS_CONFIG

def build_pr_net(cfg, num_joints, input_channels=480):
    net = PoseRefine(cfg, num_joints, input_channels)
    return net

class PoseRefine(nn.Module):
    def __init__(self, cfg, num_joints, input_channels=480):
        super(PoseRefine, self).__init__()
        self.num_joints = num_joints
        dataset = cfg.DATASET.DATASET
        if 'ochuman' in dataset:
            dataset = 'COCO'
        elif 'crowdpose' in dataset:
            dataset = 'CROWDPOSE'
        else:
            dataset = 'COCO'
        self.part_idx = VIS_CONFIG[dataset]['part_idx']
        self.part_labels = VIS_CONFIG[dataset]['part_labels']
        self.part_orders = VIS_CONFIG[dataset]['part_orders']

        self.num_layers = cfg.REFINE.NUM_LAYERS
        init_graph = self.build_graph()
        self.adj = nn.Parameter(init_graph)

        self.gconv_head = BasicGraphBlock(input_channels, input_channels)
        gconv_layers = [ResGraphBlock(input_channels, input_channels, input_channels) for _ in range(self.num_layers)]
        self.gconv_layers = nn.ModuleList(gconv_layers)
        self.gconv_pred = nn.Sequential(
            nn.Conv1d(input_channels * 3, input_channels, 1, 1, 0),
            nn.BatchNorm1d(input_channels),
            nn.ReLU(inplace=True),
            nn.Conv1d(input_channels, self.num_joints * 2, 1, 1, 0)
        )

    def forward(self, features, proposals):
        coords, center_ind = proposals
        feat_joints = float_sample(features, coords)  # batch size x max people x num_joint x feat_dim
        feat_center = int_sample(features, center_ind)
        b, num_people, num_joints, feat_dim = feat_joints.shape

        feats = feat_joints.reshape(b * num_people, num_joints, -1)

        feats = self.gconv_head(feats, self.adj)
        for i in range(self.num_layers):
            feats = self.gconv_layers[i](feats, self.adj)
        
        feat1 = torch.mean(feats, dim=1).reshape(b, num_people, feat_dim).permute(0, 2, 1)
        feat2 = torch.max(feats, dim=1)[0].reshape(b, num_people, feat_dim).permute(0, 2, 1)
        feats = torch.cat((feat1, feat2, feat_center.permute(0, 2, 1)), dim=1)
        refine_offset = self.gconv_pred(feats).permute(0, 2, 1).reshape(b, num_people, num_joints*2)

        return refine_offset

    def build_graph(self):
        graph = torch.eye(len(self.part_labels))
        for i, part in enumerate(self.part_labels):
            for (p1, p2) in self.part_orders:
                if p1 == part: graph[i, self.part_idx[p2]] = 1
                if p2 == part: graph[i, self.part_idx[p1]] = 1

        rowsum = graph.sum(1)
        r_inv_sqrt = torch.pow(rowsum, -0.5).flatten()
        r_inv_sqrt[torch.isinf(r_inv_sqrt)] = 0.
        r_mat_inv_sqrt = torch.diag(r_inv_sqrt)
        graph = r_mat_inv_sqrt.mm(graph).mm(r_mat_inv_sqrt)
        return graph