from torch import nn

def build_ppg_net(cfg, num_joints, num_det_joints, input_channels=480):
    net = PartPoseGeneration(cfg, num_joints, num_det_joints, input_channels)
    return net

class PartPoseGeneration(nn.Module):
    def __init__(self, cfg, num_joints, num_det_joints, input_channels=480):
        super(PartPoseGeneration, self).__init__()
        self.dim_heat = num_joints
        self.dim_detkpt = num_det_joints
        self.dim_reg = num_det_joints * num_joints * 2
        backbone_cfg = cfg.MODEL.BACKBONE

        self.heatmap_head = nn.Conv2d(
            in_channels=input_channels,
            out_channels=self.dim_heat,
            kernel_size=backbone_cfg.FINAL_CONV_KERNEL,
            stride=1,
            padding=1 if backbone_cfg.FINAL_CONV_KERNEL == 3 else 0
        )
        
        self.detkptmap_head = nn.Conv2d(
            in_channels=input_channels,
            out_channels=self.dim_detkpt,
            kernel_size=backbone_cfg.FINAL_CONV_KERNEL,
            stride=1,
            padding=1 if backbone_cfg.FINAL_CONV_KERNEL == 3 else 0
        )
        
        self.offset_head = nn.Conv2d(
            in_channels=input_channels,
            out_channels=self.dim_reg,
            kernel_size=backbone_cfg.FINAL_CONV_KERNEL,
            stride=1,
            padding=1 if backbone_cfg.FINAL_CONV_KERNEL == 3 else 0
        )
        

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.normal_(m.weight, std=0.001)
                for name, _ in m.named_parameters():
                    if name in ['bias']:
                        nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        heatmap = self.heatmap_head(x)
        detkptmap = self.detkptmap_head(x)
        offset = self.offset_head(x)
        return heatmap, detkptmap, offset