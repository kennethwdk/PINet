import torch
import numpy as np
from utils.transforms import up_interpolate

def aggregate_output(cfg, heatmap_sum, poses, scores, heatmap, pose_this_scale, score_this_scale, scale):
    ratio = cfg.DATASET.INPUT_SIZE * 1.0 / cfg.DATASET.OUTPUT_SIZE
    reverse_scale = ratio / scale
    h, w = heatmap[0].size(-1), heatmap[0].size(-2)
    heatmap_sum += up_interpolate(
        heatmap,
        size=(int(reverse_scale*w), int(reverse_scale*h)),
        mode='bilinear'
    )
    if len(pose_this_scale) > 0:
        poses.append(reverse_scale * pose_this_scale)
        scores.append(score_this_scale)
    else:
        poses.append([])
        scores.append([])

    return heatmap_sum, poses, scores

def adjust_output(cfg, poses, scores, heatmap):
    num_scale = len(poses)
    scale1_index = sorted(cfg.TEST.SCALE_FACTOR, reverse=True).index(1.0)
    max_score = scores[scale1_index].max() if len(scores[scale1_index]) else 1
    
    final_poses, final_scores = [], []
    for idx in range(num_scale):
        pose, score = poses[idx], scores[idx]
        if len(pose) == 0: continue
        if idx != scale1_index:
            max_score_scale = score.max() if score.shape[0] else 1
            score = score / max_score_scale * max_score * cfg.TEST.SCALE_DECREASE

        num_people, num_keypoints = pose.size()[:2]
        heatval = np.zeros((num_people, num_keypoints, 1))

        for i in range(num_people):
            for j in range(num_keypoints):
                k1, k2 = int(pose[i, j, 0]), int(pose[i, j, 0]) + 1
                k3, k4 = int(pose[i, j, 1]), int(pose[i, j, 1]) + 1
                u = pose[i, j, 0] - int(pose[i, j, 0])
                v = pose[i, j, 1] - int(pose[i, j, 1])
                if k2 < heatmap.shape[2] and k1 >= 0 and k4 < heatmap.shape[1] and k3 >= 0:
                    heatval[i, j, 0] = heatmap[j, k3, k1] * (1 - v) * (1 - u) + heatmap[j, k4, k1] * (1 - u) * v + \
                        heatmap[j, k3, k2] * u * (1 - v) + heatmap[j, k4, k2] * u * v

        score = torch.tensor(
            score[:, None].expand(-1, num_keypoints)[:, :, None].cpu().numpy() * heatval).float()
        pose = torch.cat([pose.cpu(), score.cpu()], dim=2)
        pose = pose.cpu().numpy()

        score = np.mean(pose[:, :, 2], axis=1)

        final_poses.append(pose)
        final_scores.append(score)
    
    if len(final_poses) > 0:
        final_poses = np.concatenate(final_poses, axis=0)
        final_scores = np.concatenate(final_scores, axis=0)

    return final_poses, final_scores