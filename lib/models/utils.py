import torch
from torch.nn import functional as F

def get_proposals(offsets, detkptmap, gt_inds):
    '''
    Args:
        offsets: batch size x num_joints*2 x H x W
        heatmaps: batch size x num_joints+1 x H xW
    Returns:
        kpts: batch size x max_num_people x num_joints x 2
    '''
    b, _, h, w = offsets.shape
    c = offsets.size(1) // detkptmap.size(1)
    coords_list = []
    center_ind_list = []
    pred_offsets_list = []
    for i in range(detkptmap.size(1)):
        b, _, h, w = offsets[:, i*c:(i+1)*c, :, :].shape
        coords = get_reg_kpts(offsets[:, i*c:(i+1)*c, :, :]) # batch size x HW x num_joints x 2

        pos_ind = gt_inds[i]
        coords = coords[torch.arange(b, device=offsets.device)[:, None], pos_ind]

        x = pos_ind % w
        y = (pos_ind / w).long()
        center_ind = torch.stack((y, x), dim=2)

        pred_offsets = offsets[:, i*c:(i+1)*c, :, :].clone().detach().permute(0, 2, 3, 1).reshape(b, h * w, c)
        pred_offsets = pred_offsets[torch.arange(b, device=offsets.device)[:, None], pos_ind]

        coords_list.append(coords)
        center_ind_list.append(center_ind)
        pred_offsets_list.append(pred_offsets)

    coords = torch.cat(coords_list, dim=1)
    center_ind = torch.cat(center_ind_list, dim=1)
    pred_offsets = torch.cat(pred_offsets_list, dim=1)

    return [coords, center_ind], pred_offsets

def get_reg_kpts(offsets):
    b, c, h, w = offsets.shape
    num_joints = (c // 2)
    offsets = offsets.permute(0, 2, 3, 1).reshape(b, h*w, num_joints, 2)
    shifts_x = torch.arange(0, w, step=1, dtype=torch.float32, device=offsets.device)
    shifts_y = torch.arange(0, h, step=1, dtype=torch.float32, device=offsets.device)
    shift_y, shift_x = torch.meshgrid(shifts_y, shifts_x)
    shift_x = shift_x.reshape(-1)
    shift_y = shift_y.reshape(-1)
    locations = torch.stack((shift_x, shift_y), dim=1)
    locations = locations[:, None, :].expand(-1, num_joints, -1)
    locations = locations[None, :, :, :].expand(b, -1, -1, -1)
    kpts = locations - offsets
    # kpts = torch.flip(kpts, [3])
    return kpts

def int_sample(features, pos_ind):
    feats = features[torch.arange(pos_ind.size(0), device=features.device)[:, None], :,
            pos_ind[:, :, 0], pos_ind[:, :, 1]]
    return feats

def float_sample(features, coords):
    B, C, H, W = features.size()
    trans_coords = coords / torch.tensor([(W-1)/2, (H-1)/2], dtype=torch.float32, device=features.device)[None, None,
                            None, :] - 1
    feats = F.grid_sample(features, trans_coords, align_corners=False)
    feats = feats.permute(0, 2, 3, 1).contiguous()
    return feats
