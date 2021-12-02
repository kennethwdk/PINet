import logging
import time
import torch

from loss.heatmaploss import HeatmapLoss
from loss.offsetloss import OffsetLoss
from loss.refineloss import RefineLoss

class Trainer(object):
    def __init__(self, cfg, model, rank, output_dir):
        self.model = model
        self.output_dir = output_dir
        self.rank = rank
        self.print_freq = cfg.PRINT_FREQ
        self.refine = cfg.REFINE.USE_REFINE
        self.max_num_proposal = cfg.REFINE.MAX_PROPOSAL

        self.heatmap_loss = HeatmapLoss()
        self.offset_loss = OffsetLoss()
        self.refine_loss = RefineLoss()
        self.heatmap_loss_weight = cfg.LOSS.HEATMAP_LOSS_FACTOR
        self.offset_loss_weight = cfg.LOSS.OFFSET_LOSS_FACTOR
        self.refine_loss_weight = cfg.LOSS.REFINE_LOSS_FACTOR

    def train(self, epoch, data_loader, optimizer):
        logger = logging.getLogger("Training")

        batch_time = AverageMeter()
        data_time = AverageMeter()
        heatmap_loss_meter = AverageMeter()
        offset_loss_meter = AverageMeter()
        if self.refine:
            refine_loss_meter = AverageMeter()

        self.model.train()

        end = time.time()
        for i, (images, heatmaps, kpt_masks, detkpt_maps, detkpt_masks, offsets, weights) in enumerate(data_loader):
            data_time.update(time.time() - end)

            heatmaps, kpt_masks, detkpt_maps, detkpt_masks, offsets, weights = heatmaps.cuda(non_blocking=True), kpt_masks.cuda(non_blocking=True), detkpt_maps.cuda(non_blocking=True), detkpt_masks.cuda(non_blocking=True), offsets.cuda(non_blocking=True), weights.cuda(non_blocking=True)

            gt_inds, gt_offsets, reg_weights = get_gt_proposals(heatmaps, detkpt_maps, offsets, weights, max_num_proposal=self.max_num_proposal)
            batch_inputs = {}
            batch_inputs.update({'images': images})
            batch_inputs.update({'gt_inds': gt_inds})
            outputs = self.model(batch_inputs)

            pred_heatmaps, pred_detkptmaps, pred_offsets = outputs[0:3]
            pred_heatmaps_all = torch.cat((pred_heatmaps, pred_detkptmaps), dim=1)
            gt_heatmaps = torch.cat((heatmaps, detkpt_maps), dim=1)
            gt_masks = torch.cat((kpt_masks, detkpt_masks), dim=1)

            heatmap_loss = self.heatmap_loss(pred_heatmaps_all, gt_heatmaps, gt_masks)
            offset_loss = self.offset_loss(pred_offsets, offsets, weights)
            if self.refine:
                refine_offsets = outputs[3]
                refine_loss = self.refine_loss(refine_offsets, gt_offsets, reg_weights)

            loss = 0
            if heatmap_loss is not None:
                heatmap_loss = heatmap_loss.mean(dim=0) * self.heatmap_loss_weight
                heatmap_loss_meter.update(heatmap_loss.item(), images.size(0))
                loss = loss + heatmap_loss
            if offset_loss is not None:
                offset_loss = offset_loss * self.offset_loss_weight
                offset_loss_meter.update(offset_loss.item(), images.size(0))
                loss = loss + offset_loss
            if self.refine:
                refine_loss = refine_loss * self.refine_loss_weight
                refine_loss_meter.update(refine_loss.item(), images.size(0))
                loss = loss + refine_loss

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            batch_time.update(time.time() - end)
            end = time.time()

            if i % self.print_freq == 0 and self.rank == 0:
                msg = 'Epoch: [{0}][{1}/{2}]\t' \
                      'Time: {batch_time.val:.3f}s ({batch_time.avg:.3f}s)\t' \
                      'Speed: {speed:.1f} samples/s\t' \
                      'Data: {data_time.val:.3f}s ({data_time.avg:.3f}s)\t' \
                      '{heatmaps_loss}{offset_loss}'.format(
                        epoch, i, len(data_loader),
                        batch_time=batch_time,
                        speed=images.size(0) / batch_time.val,
                        data_time=data_time,
                        heatmaps_loss=_get_loss_info(heatmap_loss_meter, 'heatmap'),
                        offset_loss=_get_loss_info(offset_loss_meter, 'offset')
                    )
                if self.refine:
                    msg += _get_loss_info(refine_loss_meter, 'refine')
                logger.info(msg)

def _get_loss_info(meter, loss_name):
    msg = '{name}: {meter.val:.3e} ({meter.avg:.3e})\t'.format(name=loss_name, meter=meter)
    return msg

def get_gt_proposals(heatmaps, detkptmaps, offsets, weights, pos_inds=None, max_num_proposal=200):
    b, c, h, w = heatmaps.shape
    num_det = detkptmaps.size(1)
    num_joints = heatmaps.size(1)
    step = num_joints * 2
    if pos_inds is None:
        pos_inds = []
        for i in range(num_det):
            offset_w = (detkptmaps[:, i:i+1, :, :] * torch.max(weights[:, i*step:(i+1)*step, :, :], dim=1, keepdim=True)[0]).view(b, -1)
            num_nonzero = (offset_w > 0).sum(1).min().item()
            if num_nonzero == 0: num_nonzero = max_num_proposal // num_det
            num_nonzero = min(max_num_proposal // num_det, num_nonzero)
            _, pos_ind = offset_w.topk(num_nonzero, dim=1)
            pos_inds.append(pos_ind)

    gt_offsets, reg_weights = [], []
    step = num_joints * 2
    for i in range(num_det):
        gt_offset = offsets[:, i*step:(i+1)*step, :, :].permute(0, 2, 3, 1).reshape(b, h * w, step)
        gt_offset = gt_offset[torch.arange(b, device=offsets.device)[:, None], pos_inds[i]]
        gt_offsets.append(gt_offset)
        reg_weight = weights[:, i*step:(i+1)*step, :, :].permute(0, 2, 3, 1).reshape(b, h * w, step)
        reg_weight = reg_weight[torch.arange(b, device=offsets.device)[:, None], pos_inds[i]]
        reg_weights.append(reg_weight)
    gt_offsets = torch.cat(gt_offsets, dim=1)
    reg_weights = torch.cat(reg_weights, dim=1)

    return pos_inds, gt_offsets, reg_weights

class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count if self.count != 0 else 0