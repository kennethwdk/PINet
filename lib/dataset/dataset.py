import logging
import os
import os.path

import cv2
import numpy as np
from torch.utils.data import Dataset
import pycocotools

from .utils import HeatmapGenerator, OffsetGenerator

logger = logging.getLogger(__name__)

class PoseDataset(Dataset):
    def __init__(self, cfg, is_train, transform=None):
        super(PoseDataset, self).__init__()
        self.root = cfg.DATASET.ROOT
        self.dataset = cfg.DATASET.DATASET
        if self.dataset == 'crowdpose':
            from crowdposetools.coco import COCO
        else:
            from pycocotools.coco import COCO
        self.split = cfg.DATASET.TRAIN if is_train else cfg.DATASET.TEST
        self.is_train = is_train
        self.transform = transform
        self.coco = COCO(self._get_anno_file_name())
        self.ids = list(self.coco.imgs.keys())

        self.num_joints = cfg.DATASET.NUM_JOINTS
        self.det_type = cfg.DATASET.DET_TYPE
        self.detkpt = cfg.DATASET.DETKPT_NAME
        self.detkpt_idxs = cfg.DATASET.DETKPT_IDXS
        if is_train:
            self.detkpt_sigma = cfg.DATASET.DETKPT_SIGMA
            self.sigma = cfg.DATASET.SIGMA
            self.bg_weight = cfg.DATASET.BG_WEIGHT
            if self.dataset == 'coco':
                self.ids = self._filter_img()
            else:
                self.ids = [img_id for img_id in self.ids if len(self.coco.getAnnIds(imgIds=img_id, iscrowd=None)) > 0]
            self.heatmap_generator = HeatmapGenerator(cfg.DATASET.OUTPUT_SIZE, cfg.DATASET.NUM_JOINTS)
            self.detkptmap_generator = HeatmapGenerator(cfg.DATASET.OUTPUT_SIZE, len(cfg.DATASET.DETKPT_NAME))
            self.offset_generator = OffsetGenerator(cfg.DATASET.OUTPUT_SIZE, cfg.DATASET.OUTPUT_SIZE,
                                                        cfg.DATASET.NUM_JOINTS, len(cfg.DATASET.DETKPT_NAME), cfg.DATASET.OFFSET_RADIUS)

    def _get_anno_file_name(self):
        return os.path.join(self.root, 'annotations', '{}_{}.json'.format(self.dataset, self.split))

    def _get_image_path(self, file_name):
        if self.dataset == 'coco':
            images_dir = os.path.join(self.root, 'images', '{}2017'.format(self.split))
        else:
            images_dir = os.path.join(self.root, 'images')
        return os.path.join(images_dir, file_name)

    def _filter_img(self):
        filtered_ids = []
        for img_id in self.ids:
            ann_id = self.coco.getAnnIds(imgIds=img_id, iscrowd=None)
            ann_info = self.coco.loadAnns(ann_id)
            num_keypoints_sum = 0
            for i, anno in enumerate(ann_info):
                num_keypoints_sum += anno['num_keypoints']
            if num_keypoints_sum > 10:
                filtered_ids.append(img_id)
        return filtered_ids

    def __getitem__(self, index):
        coco = self.coco
        img_id = self.ids[index]
        if self.dataset == 'coco' and self.split == 'test':
            file_name = coco.loadImgs(img_id)[0]['file_name']
            img = cv2.imread(
                self._get_image_path(file_name),
                cv2.IMREAD_COLOR | cv2.IMREAD_IGNORE_ORIENTATION
            )
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            return img, img_id
        ann_ids = coco.getAnnIds(imgIds=img_id)
        target = coco.loadAnns(ann_ids)
        anno = [obj for obj in target]
        file_name = coco.loadImgs(img_id)[0]['file_name']
        img = cv2.imread(
            self._get_image_path(file_name),
            cv2.IMREAD_COLOR | cv2.IMREAD_IGNORE_ORIENTATION
        )
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        if self.is_train:
            return self.process_train_image(img, anno, index)
        else:
            return self.process_test_image(img, anno, img_id)

    def __len__(self):
        return len(self.ids)

    def process_train_image(self, img, anno, idx):
        img_info = self.coco.loadImgs(self.ids[idx])[0]
        m = np.zeros((img_info['height'], img_info['width']))
        if self.dataset == 'coco':
            for obj in anno:
                if obj['iscrowd']:
                    rle = pycocotools.mask.frPyObjects(
                        obj['segmentation'], img_info['height'], img_info['width'])
                    m += pycocotools.mask.decode(rle)
                elif obj['num_keypoints'] == 0:
                    rles = pycocotools.mask.frPyObjects(
                        obj['segmentation'], img_info['height'], img_info['width'])
                    for rle in rles:
                        m += pycocotools.mask.decode(rle)
        mask = m < 0.5

        anno = [obj for obj in anno if obj['iscrowd'] == 0 or obj['num_keypoints'] > 0]
        num_people = len(anno)
        area = np.zeros((num_people, 1))
        joints = np.zeros((num_people, self.num_joints, 3))
        det_joints = np.zeros((num_people, len(self.detkpt), 3))

        for i, obj in enumerate(anno):
            joints[i, :, :3] = np.array(obj['keypoints']).reshape([-1, 3])
            area[i, 0] = obj['bbox'][2] * obj['bbox'][3]

        if self.transform:
            img, mask, joints, area = self.transform(img, mask, joints, area)

        if len(self.detkpt) > 0:
            for i, obj in enumerate(anno):
                if not self.dataset == 'crowdpose':
                    if area[i, 0] < 32 ** 2:
                        det_joints[i, :, 2] = 0
                        continue
                for j, det_idx in enumerate(self.detkpt_idxs):
                    joints_sel = joints[i, det_idx, :2]
                    vis = (joints[i, det_idx, 2:3] > 0).astype(np.float32)
                    joints_sum = np.sum(joints_sel * vis, axis=0)
                    num_vis_joints = len(np.nonzero(joints[i, det_idx, 2])[0])
                    vis_joints = np.sum(joints[i, det_idx, 2])
                    if num_vis_joints <= 0:
                        det_joints[i, j, 2] = 0
                    else:
                        det_joints[i, j, :2] = joints_sum / num_vis_joints
                        det_joints[i, j, 2] = 2

        heatmap, ignored = self.heatmap_generator(joints, self.sigma, self.bg_weight)
        kpt_mask = (mask * ignored).astype(np.float32)

        detkpt_map, ignored = self.detkptmap_generator(det_joints, self.detkpt_sigma, self.bg_weight)
        detkpt_mask = (mask * ignored).astype(np.float32)

        offset, weight = self.offset_generator(joints, det_joints, area)
        
        return img, heatmap, kpt_mask, detkpt_map, detkpt_mask, offset, weight

    def process_test_image(self, img, anno, img_id):
        img_h, img_w = img.shape[:2]
        num_people = len(anno)
        area = np.zeros((num_people, 1))
        joints = np.zeros((num_people, self.num_joints, 3))
        det_joints = np.zeros((num_people, len(self.detkpt), 3))

        for i, obj in enumerate(anno):
            joints[i, :, :3] = np.array(obj['keypoints']).reshape([-1, 3])
            area[i, 0] = obj['bbox'][2] * obj['bbox'][3]
        
        if len(self.detkpt) > 0:
            for i, obj in enumerate(anno):
                for j, det_idx in enumerate(self.detkpt_idxs):
                    joints_sel = joints[i, det_idx, :2]
                    vis = (joints[i, det_idx, 2:3] > 0).astype(np.float32)
                    joints_sum = np.sum(joints_sel * vis, axis=0)
                    num_vis_joints = len(np.nonzero(joints[i, det_idx, 2])[0])
                    vis_joints = np.sum(joints[i, det_idx, 2])
                    if num_vis_joints <= 0:
                        det_joints[i, j, 2] = 0
                    else:
                        det_joints[i, j, :2] = joints_sum / num_vis_joints
                        det_joints[i, j, 2] = 2

        human_mask = np.zeros((img_h, img_w))
        for _, obj in enumerate(anno):
            box = obj['bbox']
            tl_x = int(box[0] - 0.5)
            tl_y = int(box[1] - 0.5)
            br_x = int(box[0] + box[2] + 0.5)
            br_y = int(box[1] + box[3] + 0.5)
            human_mask[tl_y:br_y, tl_x:br_x] = 1

        return img, img_id, joints, det_joints, human_mask, area

    def __repr__(self):
        fmt_str = 'Dataset ' + self.__class__.__name__ + '\n'
        fmt_str += '    Number of datapoints: {}\n'.format(self.__len__())
        fmt_str += '    Root Location: {}'.format(self.root)
        return fmt_str
