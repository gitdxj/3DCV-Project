# The code is adapted from https://github.com/j96w/DenseFusion/blob/master/datasets/linemod/dataset.py

import os
import torch
import numpy as np
from PIL import Image
import torchvision.transforms as transforms
import torch.utils.data as data
import yaml


class LinemodDataset(data.Dataset):
    def __init__(self, mode, dataset_path, cloud_pt_num, obj_list=[1, 2, 4, 5, 6, 8, 9, 10, 11, 12, 13, 14, 15]):
        self.mode = mode
        self.objects = obj_list
        self.cloud_pt_num = cloud_pt_num
        self.sys_obj = [7, 8]

        self.rgb_list = []        # path to rgb img
        self.depth_list = []      # path to depth img
        self.label_list = []      # path to mask
        self.obj_list = []        # obj index: 1 to 15
        self.index_list = []      # img index
        # self.diameter_dict = dict()   # model diameters
        self.gt_dict = dict()          # ground truth: rotation, translation and bb
        self.vtx_dict = dict()         # vertexes of models read from ply file

        model_info_path = os.path.join(dataset_path, 'models_info.yml')
        self.model_info_dict = yaml.load(open(model_info_path), Loader=yaml.CLoader)   # key: obj_index, val: dict

        for obj in obj_list:
            if mode == 'train':
                index_file = os.path.join(dataset_path, 'data', str(obj), 'train.txt')
            else:
                index_file = os.path.join(dataset_path, 'data', str(obj), 'test.txt')
            file = open(index_file)
            index_list = [line.rstrip() for line in file]   # get rid of '\n' in each line, elements are of str

            gt_path = os.path.join(dataset_path, 'data', str(obj), 'gt.yml')
            info_path = os.path.join(dataset_path, 'data', str(obj), 'info.yml')
            ply_path = os.path.join(dataset_path, 'models', 'obj_' + str(obj) + '.ply')

            # read vertexes of model from ply file
            vtx = read_ply_vtx(ply_path)
            self.vtx_dict[obj] = vtx
            # read ground truth to gt_dict
            self.gt_dict[obj] = yaml.load(gt_path, Loader=yaml.CLoader)

            for index in index_list:
                rgb_path = os.path.join(dataset_path, 'data', 'rgb', index+'.png')
                depth_path = os.path.join(dataset_path, 'data', 'depth', index+'.png')
                if mode == 'eval':
                    label_path = os.path.join(dataset_path, 'segnet_results', str(obj)+'_label', index+'_label.png')
                else:
                    label_path = os.path.join(dataset_path, 'data', str(obj), 'mask', index+'.png')

                self.obj_list.append(obj)
                self.index_list.append(int(index))
                self.rgb_list.append(rgb_path)
                self.depth_list.append(depth_path)
                self.label_list.append(label_path)

        self.cam_cx = 325.26110
        self.cam_cy = 242.04899
        self.cam_fx = 572.41140
        self.cam_fy = 573.57043

        self.xmap = np.array([[i for i in range(640)] for j in range(480)])
        self.ymap = np.array([[j for i in range(640)] for j in range(480)])

        self.num_pt_mesh = 500

        self.trancolor = transforms.ColorJitter(0.2, 0.2, 0.2, 0.05)
        self.transform = transforms.Compose([transforms.ToTensor(),
                                             transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                                                  std=[0.229, 0.224, 0.225])])

    def __getitem__(self, item):
        # read rgb img, depth img and mask
        rgb_img = np.array(Image.open(self.rgb_list[item]))
        depth_img = np.array(Image.open(self.depth_list[item]))
        label_img = np.array(Image.open(self.label_list[item]))

        obj_id = self.obj_list[item]
        img_id = self.index_list[item]

        # get ground truth parameters from self.gt_dict
        if obj_id == 2:
            for each_dict in self.gt_dict[obj_id][img_id]:
                if each_dict['obj_id'] == 2:
                    gt = each_dict
                    break
        else:
            gt = self.gt_dict[obj_id][img_id][0]

        # output a point cloud
        depth_mask = np.ma.getmaskarray(np.ma.masked_not_equal(depth_img, 0))    # mask non-zero as True
        if self.mode == 'eval':
            label_mask = np.ma.getmaskarray(np.ma.masked_equal(label_img, np.array(255)))   # mask 255 as True
        else:
            # mask 255，255，255 as True, since the img has 3 channel
            label_mask = np.ma.getmaskarray(np.ma.masked_equal(label_img, np.array([255, 255, 255])))[:, :, 0]
        mask = label_mask * depth_mask

        # get the bounding box and crop img
        rmin, rmax, cmin, cmax = get_bb(gt['obj_bb'])
        img_crop = rgb_img[rmin:rmax, cmin:cmax, :]

        # select cloud_pt_num points to make a point cloud
        choice = mask[rmin:rmax, cmin:cmax].flatten().nonzero()[0]  # nonzero returns a tuple
        if len(choice) == 0:
            return None, None, None, None, None, None
        if len(choice) > self.cloud_pt_num:
            # randomly select cloud_pt_num points
            choice = np.random.choice(choice, self.cloud_pt_num, replace=False)
        # repeat points if not enough
        else:
            choice = np.pad(choice, (0, self.cloud_pt_num - len(choice)), 'wrap')

        depth_points = depth_img[rmin:rmax, cmin:cmax].flatten()[choice].reshape(-1, 1)
        x_points = self.xmap[rmin:rmax, cmin:cmax].flatten()[choice].reshape(-1, 1)
        y_points = self.ymap[rmin:rmax, cmin:cmax].flatten()[choice].reshape(-1, 1)
        choice = np.array([choice])

        pt2 = depth_points / 1000
        pt0 = (x_points - self.cam_cx) * pt2 / self.cam_fx
        pt1 = (y_points - self.cam_cy) * pt2 / self.cam_fy
        cloud = np.hstack((pt0, pt1, pt2))

        # get ground truth rotation and translation
        target_rotation = np.resize(np.array(gt['cam_R_m2c']), (3, 3))
        target_translation = np.array(gt['cam_t_m2c']) / 1000

        gt_translation = target_translation
        target_translation -= cloud
        target_translation /= np.linalg.norm(target_translation, axis=1)
        target_translation = np.array([target_translation])

        model_vtx = self.vtx_dict[obj_id] / 1000.0
        vtx_choice = np.random.choice(len(model_vtx), self.num_pt_mesh, replace=False)
        model_vtx = model_vtx[vtx_choice, :]
        target_rotation = np.dot(model_vtx, target_rotation.T)

        return (torch.from_numpy(cloud.astype(np.float32)),
                torch.LongTensor(choice.astype(np.int32)),
                self.transform(img_crop),
                torch.from_numpy(target_translation.astype(np.float32)),
                torch.from_numpy(target_rotation.astype(np.float32)),
                torch.from_numpy(model_vtx.astype(np.float32)),
                torch.LongTensor([self.objects.index(obj_id)]),
                torch.from_numpy(gt_translation.astype(np.float32)))

    def __len__(self):
       return len(self.rgb_list)


def read_ply_vtx(filepath):
    """
    get the vertexes of a model
    :param filepath: the path of ply file
    :return: an array of size (num_vtx, 3)
    """
    f = open(filepath)
    assert f.readline().strip() == "ply"
    f.readline()
    f.readline()
    N = int(f.readline().split()[-1])
    while f.readline().strip() != "end_header":
        continue
    pts = []
    for _ in range(N):
        pts.append(np.float32(f.readline().split()[:3]))
    return np.array(pts)


def get_bb(bb):
   pass