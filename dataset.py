import os
from PIL import image
import torchvision.transforms as transforms
import torch.utils.data as data


class LinemodDataset(data.Dataset):
    def __init__(self, mode, dataset_path, obj_list=[1, 2, 4, 5, 6, 8, 9, 10, 11, 12, 13, 14, 15]):
        self.mode = mode

        self.rgb_list = []     # path to rgb img
        self.depth_list = []   # path to depth img
        self.label_list = []   # path to mask
        self.obj_list = []     # obj number
        self.index_list = []   # data index

        for obj in obj_list:
            if mode == 'train':
                index_file = os.path.join(dataset_path, 'data', str(obj), 'train.txt')
            else:
                index_file = os.path.join(dataset_path, 'data', str(obj), 'test.txt')
            file = open(index_file)
            index_list = [line.rstrip() for line in file]   # get rid of '\n' in each line, elements are of str

            for index in index_list:
                rgb_path = os.path.join(dataset_path, 'data', 'rgb', index+'.png')
                depth_path = os.path.join(dataset_path, 'data', 'depth', index+'.png')
                if mode == 'eval':
                    label_path = os.path.join(dataset_path, 'segnet_results', str(obj)+'_label', index+'_label.png')
                else:
                    label_path = os.path.join(dataset_path, 'data', str(obj), 'mask', index+'.png')
                gt_path = os.path.join(dataset_path, 'data', str(obj), 'gt.yml')
                info_path = os.path.join(dataset_path, 'data', str(obj), 'info.yml')
                ply_path = os.path.join(dataset_path, 'models', 'obj_'+str(obj)+'.ply')

                self.obj_list.append(obj)
                self.index_list.append(int(index))
                self.rgb_list.append(rgb_path)
                self.depth_list.append(depth_path)
                self.label_list.append(label_path)





    def __getitem__(self, item):
        pass

    def __len__(self):
        pass