import torch
from torch import nn
from torch.nn import functional as F

import logging
logger = logging.getLogger(__name__)

from .utils import get_proposals
from .backbone import build_backbone
from .ppg import build_ppg_net
from .pr import build_pr_net

from dataset.utils import FLIP_CONFIG

class PINet(nn.Module):
    def __init__(self, cfg, is_train, **kwargs):
        super(PINet, self).__init__()
        self.num_joints = cfg.DATASET.NUM_JOINTS
        self.num_detkpt_joints = len(cfg.DATASET.DETKPT_NAME)
        self.output_channels = cfg.MODEL.BACKBONE.OUTPUT_DIM
        self.refine = cfg.REFINE.USE_REFINE

        # inference
        self.flip_test = cfg.TEST.FLIP_TEST
        if 'coco' in cfg.DATASET.DATASET:
            self.flip_index = FLIP_CONFIG['COCO']
        elif 'crowdpose' in cfg.DATASET.DATASET:
            self.flip_index = FLIP_CONFIG['CROWDPOSE']
        elif 'ochuman' in cfg.DATASET.DATASET:
            self.flip_index = FLIP_CONFIG['OCHUMAN']
        self.max_proposals = cfg.DATASET.MAX_NUM_PEOPLE
        self.keypoint_thre = cfg.TEST.KEYPOINT_THRESHOLD
        self.pool_thre1 = cfg.TEST.POOL_THRESHOLD1
        self.pool_thre2 = cfg.TEST.POOL_THRESHOLD2

        self.backbone = build_backbone(cfg, is_train=is_train)
        self.ppg = build_ppg_net(cfg, self.num_joints, self.num_detkpt_joints, self.output_channels)
        if self.refine:
            self.pr = build_pr_net(cfg, self.num_joints, self.output_channels)

    def forward(self, batch_inputs):
        images = batch_inputs['images']
        features = self.backbone(images)
        heatmap, detkptmap, offset = self.ppg(features)

        if self.training:
            ret = []
            ret.extend([heatmap, detkptmap, offset])
            gt_inds = batch_inputs['gt_inds']
            if self.refine:
                with torch.no_grad():
                    proposals, pred_offsets = get_proposals(offset.clone().detach(), detkptmap.clone().detach(), gt_inds)
                refine_offsets = self.pr(features, proposals)
                refine_offsets = refine_offsets + pred_offsets
                ret.append(refine_offsets)
            return ret
        else:
            images_flip = torch.zeros_like(images)
            images_tmp = torch.flip(images, [3])
            images_flip[:, :, :, :-3] = images_tmp[:, :, :, 3:]
            inp = torch.cat((images, images_flip), dim=0)
            features = self.backbone(inp)
            heatmaps, detkptmaps, offsets = self.ppg(features)
            heatmaps, poses, scores = self._forward_test(features, heatmaps, detkptmaps, offsets)
            return heatmaps, poses, scores
    
    def _forward_test(self, features, heatmaps, detkptmaps, offsets):
        B, _, H, W = detkptmaps.size()
        step = self.num_joints * 2
        offsets = offsets.reshape(B, self.num_detkpt_joints, step, H, W)

        if self.flip_test:
            heatmap_ori, heatmap_flip = heatmaps[0:1], heatmaps[1:2]
            heatmap_flip = torch.flip(heatmap_flip, [3])
            heatmaps = (heatmap_ori + heatmap_flip[:, self.flip_index, :, :]) / 2.0
            detkptmap_ori, detkptmap_flip = detkptmaps[0:1], detkptmaps[1:2]
            detkptmaps = (detkptmap_ori + torch.flip(detkptmap_flip, [3])) / 2.0
        
        poses, scores = [], []
        for i in range(self.num_detkpt_joints):
            detmap = detkptmaps[:, i, :, :]
            maxm = self.hierarchical_pool(detmap)
            maxm = torch.eq(maxm, detmap).float()
            detmap = detmap * maxm
            score = detmap.view(-1)
            score, pos_ind = score.topk(self.max_proposals, dim=0)
            select_ind = (score > (self.keypoint_thre)).nonzero()
            if len(select_ind) > 0:
                score = score[select_ind].squeeze(1)
                pos_ind = pos_ind[select_ind].squeeze(1)
                assert pos_ind.ndim == 1
                assert score.ndim == 1
                x = pos_ind % W
                y = (pos_ind / W).long()
                instance_coord = torch.stack((y, x), dim=1)
                instance_real_coord = torch.stack((x, y), dim=1)

                instance_offset = self._sample_feats(offsets[0, i], instance_coord)
                instance_pose = instance_real_coord[:, None, :] - instance_offset.reshape(-1, self.num_joints, 2)
                if self.refine:
                    refine_offset = self.pr(features[0:1], (instance_pose.unsqueeze(0), instance_coord.unsqueeze(0)))
                    instance_pose = instance_pose - refine_offset[0].reshape(-1, self.num_joints, 2)
                if self.flip_test:
                    instance_coord[:, 1] = W - instance_coord[:, 1] - 1
                    instance_real_coord[:, 0] = W - instance_real_coord[:, 0] - 1
                    instance_offset_flip = self._sample_feats(offsets[1, i], instance_coord)
                    instance_pose_flip = instance_real_coord[:, None, :] - instance_offset_flip.reshape(-1, self.num_joints, 2)
                    if self.refine:
                        refine_offset_flip = self.pr(features[1:2], (instance_pose_flip.unsqueeze(0), instance_coord.unsqueeze(0)))
                        instance_pose_flip = instance_pose_flip - refine_offset_flip[0].reshape(-1, self.num_joints, 2)
                    instance_pose_flip = instance_pose_flip[:, self.flip_index, :]
                    instance_pose_flip[:, :, 0] = W - instance_pose_flip[:, :, 0] - 1

                    instance_pose = (instance_pose + instance_pose_flip) / 2.0
                poses.append(instance_pose)
                scores.append(score)
        
        if len(poses) > 0:
            poses = torch.cat(poses, dim=0)
            scores = torch.cat(scores, dim=0)

        return heatmaps, poses, scores
    
    def _sample_feats(self, features, pos_ind):
        feats = features[:, pos_ind[:, 0], pos_ind[:, 1]]
        return feats.permute(1, 0)

    def hierarchical_pool(self, heatmap):
        map_size = (heatmap.shape[1] + heatmap.shape[2]) / 2.0
        if map_size > self.pool_thre1:
            maxm = F.max_pool2d(heatmap, 7, 1, 3)
        elif map_size > self.pool_thre2:
            maxm = F.max_pool2d(heatmap, 5, 1, 2)
        else:
            maxm = F.max_pool2d(heatmap, 3, 1, 1)
        return maxm

