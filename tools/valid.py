import argparse
import os
import sys
import stat
import pprint
from multiprocessing import Process, Queue
from collections import OrderedDict

import cv2
import numpy as np
import pickle

import torch
import torch.backends.cudnn as cudnn
import torch.nn.parallel
import torch.optim
import torch.utils.data
import torch.utils.data.distributed
import torchvision.transforms
import torch.multiprocessing
from tqdm import tqdm

import _init_paths
import models
from config import get_cfg, update_config
from core.evaluator import Evaluator
from core.inference import aggregate_output, adjust_output
from dataset import make_test_dataloader
from utils.logging import create_logger, setup_logger
from utils.transforms import resize_align_multi_scale
from utils.transforms import get_final_preds
from utils.transforms import get_multi_scale_size
from utils.nms import oks_nms, pose_fusion

torch.multiprocessing.set_sharing_strategy('file_system')

def parse_args():
    parser = argparse.ArgumentParser(description='Test DRNet')
    # general
    parser.add_argument('--cfg',
                        help='experiment configure file name',
                        required=True,
                        type=str)
    parser.add_argument('opts',
                        help="Modify config options using the command-line",
                        default=None,
                        nargs=argparse.REMAINDER)
    parser.add_argument('--gpus',
                        help='gpu ids for eval',
                        default='0',
                        type=str)
    parser.add_argument('--pf',
                        default=True,
                        type=bool)
    args = parser.parse_args()
    return args

# markdown format output
def _print_name_value(logger, name_value, full_arch_name):
    names = name_value.keys()
    values = name_value.values()
    num_values = len(name_value)
    logger.info(
        '| Arch ' +
        ' '.join(['| {}'.format(name) for name in names]) +
        ' |'
    )
    logger.info('|---' * (num_values + 1) + '|')

    if len(full_arch_name) > 15:
        full_arch_name = full_arch_name[:8] + '...'
    logger.info(
        '| ' + full_arch_name + ' ' +
        ' '.join(['| {:.3f}'.format(value) for value in values]) +
        ' |'
    )

def worker(gpu_id, dataset, indices, cfg, logger, final_output_dir, pred_queue, gpu_list, pf=False):
    os.environ['CUDA_VISIBLE_DEVICES'] = gpu_list[gpu_id]

    model = models.create(cfg.MODEL.NAME, cfg, is_train=False)

    if cfg.TEST.MODEL_FILE:
        logger.info("=> loading model from {}".format(cfg.TEST.MODEL_FILE))
        model.load_state_dict(torch.load(cfg.TEST.MODEL_FILE), strict=True)
    else:
        model_state_file = os.path.join(final_output_dir, "model_best.pth.tar")
        logger.info("=> loading model from {}".format(model_state_file))
        model.load_state_dict(torch.load(model_state_file))

    model = torch.nn.DataParallel(model).cuda()
    model.eval()

    sub_dataset = torch.utils.data.Subset(dataset, indices)
    data_loader = torch.utils.data.DataLoader(
        sub_dataset, sampler=None, batch_size=1, shuffle=False, num_workers=0, pin_memory=False
    )
    transforms = torchvision.transforms.Compose(
        [torchvision.transforms.ToTensor(),
         torchvision.transforms.Normalize(
             mean=[0.485, 0.456, 0.406],
             std=[0.229, 0.224, 0.225]
         )])

    all_preds = []

    pbar = tqdm(total=len(sub_dataset)) if cfg.TEST.LOG_PROGRESS else None
    for i, batch_inputs in enumerate(data_loader):
        (images, img_id, joints, det_joints, masks, areas) = batch_inputs
        image = images[0].cpu().numpy()
        img_id = img_id.item()
        # size at scale 1.0
        base_size, center, scale = get_multi_scale_size(
            image, cfg.DATASET.INPUT_SIZE, 1.0, 1.0
        )

        with torch.no_grad():
            heatmap_sum = 0
            poses = []
            scores = []

            for s in sorted(cfg.TEST.SCALE_FACTOR, reverse=True):
                image_resized, center, scale = resize_align_multi_scale(image, cfg.DATASET.INPUT_SIZE, s, 1.0)
                image_resized = transforms(image_resized)
                image_resized = image_resized.unsqueeze(0).cuda()
                
                heatmap_this_scale, pose_this_scale, score_this_scale = model({'images': image_resized})
                heatmap_sum, poses, scores = aggregate_output(cfg, heatmap_sum, poses, scores, heatmap_this_scale, pose_this_scale, score_this_scale, s)
            
            heatmap_avg = heatmap_sum/len(cfg.TEST.SCALE_FACTOR)
            
            poses, scores = adjust_output(cfg, poses, scores, heatmap_avg[0])

            if len(poses) == 0:
                if cfg.TEST.LOG_PROGRESS:
                    pbar.update()
                continue
            poses = get_final_preds(poses, center, scale, [heatmap_avg.size(-1), heatmap_avg.size(-2)])
            # perform nms
            keep, keep_ind = oks_nms(poses, scores, cfg.TEST.OKS_SCORE, np.array(cfg.TEST.OKS_SIGMAS) / 10.0)

            if pf:
                fused_poses, fused_scores = pose_fusion(poses, keep, keep_ind)
                for i in range(fused_poses.shape[0]):
                    all_preds.append({
                        "keypoints": fused_poses[i][:, :3].reshape(-1, ).astype(float).tolist(),
                        "image_id": img_id,
                        "score": float(fused_scores[i]),
                        "category_id": 1
                    })
            else:
                for _keep in keep:
                    all_preds.append({
                        "keypoints": poses[_keep][:, :3].reshape(-1, ).astype(float).tolist(),
                        "image_id": img_id,
                        "score": float(scores[_keep]),
                        "category_id": 1
                    })

        if cfg.TEST.LOG_PROGRESS:
            pbar.update()

    if cfg.TEST.LOG_PROGRESS:
        pbar.close()
    pred_queue.put_nowait(all_preds)

def main():
    args = parse_args()
    cfg = get_cfg()
    update_config(cfg, args)

    if args.gpus is not None:
        os.environ["CUDA_VISIBLE_DEVICES"] = args.gpus

    final_output_dir = create_logger(cfg, args.cfg, 'valid')
    logger, _ = setup_logger(final_output_dir, 0, 'valid')
    logger.info(pprint.pformat(args))
    logger.info(cfg)

    # cudnn related setting
    cudnn.benchmark = cfg.CUDNN.BENCHMARK
    torch.backends.cudnn.deterministic = cfg.CUDNN.DETERMINISTIC
    torch.backends.cudnn.enabled = cfg.CUDNN.ENABLED

    data_loader, test_dataset = make_test_dataloader(cfg)

    total_size = len(test_dataset)
    pred_queue = Queue(100)
    workers = []
    gpu_list = args.gpus.split(',')
    num_gpus = len(gpu_list)
    for i in range(num_gpus):
        indices = list(range(i, total_size, num_gpus))
        p = Process(
            target=worker,
            args=(
                i, test_dataset, indices, cfg, logger, final_output_dir, pred_queue, gpu_list, args.pf
            )
        )
        p.start()
        workers.append(p)
        logger.info("==>" + " Worker {} Started, responsible for {} images".format(i, len(indices)))

    all_preds = []
    for idx in range(num_gpus):
        all_preds += pred_queue.get()

    for p in workers:
        p.join()

    evaluator = Evaluator(cfg, final_output_dir)
    info_str = evaluator.evaluate(all_preds)
    name_values = OrderedDict(info_str)

    if isinstance(name_values, list):
        for name_value in name_values:
            _print_name_value(logger, name_value, cfg.MODEL.NAME)
    else:
        _print_name_value(logger, name_values, cfg.MODEL.NAME)


if __name__ == '__main__':
    main()
