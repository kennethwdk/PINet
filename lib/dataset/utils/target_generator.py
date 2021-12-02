import numpy as np

class HeatmapGenerator():
    def __init__(self, output_res, num_joints):
        self.output_res = output_res
        self.num_joints = num_joints

    def get_heat_val(self, sigma, x, y, x0, y0):
        g = np.exp(- ((x - x0) ** 2 + (y - y0) ** 2) / (2 * sigma ** 2))
        return g

    def __call__(self, joints, sgm, bg_weight=1.0):
        assert self.num_joints == joints.shape[1], \
            'the number of joints should be %d' % self.num_joints

        hms = np.zeros((self.num_joints, self.output_res, self.output_res),
                       dtype=np.float32)
        ignored_hms = 2*np.ones((self.num_joints, self.output_res, self.output_res),
                                dtype=np.float32)

        hms_list = [hms, ignored_hms]

        for p in joints:
            for idx, pt in enumerate(p):
                sigma = sgm
                if pt[2] > 0:
                    x, y = pt[0], pt[1]
                    if x < 0 or y < 0 or \
                            x >= self.output_res or y >= self.output_res:
                        continue

                    ul = int(np.floor(x - 3 * sigma - 1)
                             ), int(np.floor(y - 3 * sigma - 1))
                    br = int(np.ceil(x + 3 * sigma + 2)
                             ), int(np.ceil(y + 3 * sigma + 2))

                    cc, dd = max(0, ul[0]), min(br[0], self.output_res)
                    aa, bb = max(0, ul[1]), min(br[1], self.output_res)

                    joint_rg = np.zeros((bb - aa, dd - cc))
                    for sy in range(aa, bb):
                        for sx in range(cc, dd):
                            joint_rg[sy - aa, sx -
                                     cc] = self.get_heat_val(sigma, sx, sy, x, y)

                    hms_list[0][idx, aa:bb, cc:dd] = np.maximum(
                        hms_list[0][idx, aa:bb, cc:dd], joint_rg)
                    hms_list[1][idx, aa:bb, cc:dd] = 1.

        hms_list[1][hms_list[1] == 2] = bg_weight

        return hms_list

class OffsetGenerator():
    def __init__(self, output_h, output_w, num_joints, num_det_joints, radius):
        self.num_joints = num_joints
        self.output_w = output_w
        self.output_h = output_h
        self.num_joints = num_joints
        self.num_det_joints = num_det_joints
        self.radius = radius

    def __call__(self, joints, det_joints, area):
        assert joints.shape[1] == self.num_joints, \
            'the number of joints should be 17 keypoints.'

        offset_map = np.zeros((self.num_det_joints*self.num_joints*2, self.output_h, self.output_w),
                              dtype=np.float32)
        weight_map = np.zeros((self.num_det_joints*self.num_joints*2, self.output_h, self.output_w),
                              dtype=np.float32)
        area_map = np.zeros((self.num_det_joints, self.output_h, self.output_w),
                            dtype=np.float32)

        for person_id, (p, det_p) in enumerate(zip(joints, det_joints)):
            for ct_idx, ct_pt in enumerate(det_p):
                x, y, vis = ct_pt[0], ct_pt[1], ct_pt[2]
                ct_x = int(x)
                ct_y = int(y)
                ct_v = int(vis)
                if ct_v < 2 or ct_x < 0 or ct_y < 0 or ct_x >= self.output_w or ct_y >= self.output_h: continue
                st = ct_idx * self.num_joints * 2
                for idx, pt in enumerate(p):
                    if pt[2] > 0:
                        x, y = pt[0], pt[1]
                        if x < 0 or y < 0 or x >= self.output_w or y >= self.output_h: continue

                        start_x = max(int(ct_x - self.radius), 0)
                        start_y = max(int(ct_y - self.radius), 0)
                        end_x = min(int(ct_x + self.radius), self.output_w)
                        end_y = min(int(ct_y + self.radius), self.output_h)

                        for pos_x in range(start_x, end_x):
                            for pos_y in range(start_y, end_y):
                                offset_x = pos_x - x
                                offset_y = pos_y - y
                                if offset_map[st + idx*2, pos_y, pos_x] != 0 \
                                        or offset_map[st + idx*2+1, pos_y, pos_x] != 0:
                                    if area_map[ct_idx, pos_y, pos_x] < area[person_id]:
                                        continue
                                offset_map[st + idx*2, pos_y, pos_x] = offset_x
                                offset_map[st + idx*2+1, pos_y, pos_x] = offset_y
                                weight_map[st + idx*2, pos_y, pos_x] = 1. / np.sqrt(area[person_id])
                                weight_map[st + idx*2+1, pos_y, pos_x] = 1. / np.sqrt(area[person_id])
                                area_map[ct_idx, pos_y, pos_x] = area[person_id]

        return offset_map, weight_map
