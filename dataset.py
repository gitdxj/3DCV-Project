import os
import numpy as np
from PIL import Image
import torchvision.transforms as transforms
import torch.utils.data as data
import yaml


class LinemodDataset(data.Dataset):
    def __init__(self, mode, dataset_path, obj_list=[1, 2, 4, 5, 6, 8, 9, 10, 11, 12, 13, 14, 15]):
        self.mode = mode
        self.objects = obj_list
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
        rgb_img = Image.open(self.rgb_list[item])
        depth_img = Image.open(self.depth_list[item])
        label_img = Image.open(self.label_list[item])

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



        pass


    def __len__(self):
       return len(self.rgb_list)


def read_ply_vtx(filepath):
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