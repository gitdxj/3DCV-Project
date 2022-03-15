# The code is adapted from https://github.com/j96w/DenseFusion/blob/master/datasets/linemod/dataset.py

import os
import errno
import random

import numpy as np
import numpy.ma as ma
import yaml
import pandas as pd
from PIL import Image
import torch
import torchvision.transforms as transforms
from torch.utils.data import Dataset


class LinemodDataset(Dataset):
    def __init__(self, mode, add_noise, num_points, num_points_model=500, noise_trans=None,
                 dataset_dir="data/Linemod_preprocessed/"):

        # mode is either "train", "test" or "eval". train and test mode use the train/test images from the
        # original linemod dataset, whereas eval mode uses the results from the SegNet
        if mode in ["train", "test", "eval"]:
            self.mode = mode
        else:
            raise ValueError("invalid mode specified, mode has to be train, test or eval")

        # whether to add noise to the image, the translation vector and the object point cloud
        self.add_noise = add_noise
        if self.add_noise and (noise_trans is None or noise_trans <= 0):
            raise ValueError("noise transformation has to be greater than 0")
        self.noise_trans = noise_trans

        # transformations applied to the image
        # only applied if noise is added
        self.trancolor = transforms.ColorJitter(0.2, 0.2, 0.2, 0.05)
        # always applied
        self.transform = transforms.Compose([transforms.ToTensor(),
                                             transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                                                  std=[0.229, 0.224, 0.225])])

        if num_points <= 0 or num_points_model <= 0:
            raise ValueError("number of points has to be greater than 0")
        # number of points sampled for the object
        self.num_points = num_points
        # number of points sampled for the object model
        self.num_points_model = num_points_model

        # path to the directory where the data is stored
        self.dataset_dir = dataset_dir
        if not os.path.isdir(self.dataset_dir):
            raise FileNotFoundError(errno.ENOENT, os.strerror(errno.ENOENT), dataset_dir)

        # camera parameters:
        # (from the camera.json from the original linemod dataset (https://bop.felk.cvut.cz/datasets/))
        # these are the same for all images
        self.cx = 325.2611
        self.cy = 242.04899
        # depth_scale multiplied with the depth image gives depth in mm
        self.depth_scale = 1.0
        self.fx = 572.4114
        self.fy = 573.57043
        self.height = 480
        self.width = 640

        # there are 13 objects in the available linemod dataset
        self.objects = [1, 2, 4, 5, 6, 8, 9, 10, 11, 12, 13, 14, 15]
        # ids of the symmetric objects
        self.symmetric_objects = [7, 8]

        # all object ids (list with as many entries as there are images)
        self.obj_ids = []
        # images are named with numbers, this contains all numbers of the available images
        self.image_ids = []

        self.model_to_camera_rotation = []  # rotation matrix from model to camera coordinates
        self.model_to_camera_translation = []  # translation from model to camera coordinates
        # bounding boxes (x_min, x_max, y_min, y_max)
        self.obj_bb = []

        # vertex points of each model, keys are the object id
        self.model_points = {}

        # diameter of each object model
        self.model_diameter = {}
        with open(os.path.join(self.dataset_dir, "models/models_info.yml"), "r") as f:
            model_info = yaml.safe_load(f)
            for obj_id in self.objects:
                # divide by 1000 to get meters as unit
                self.model_diameter[obj_id] = model_info[obj_id]["diameter"] / 1000

        if self.mode == "train":
            get_split_image_ids_path = lambda object_id: (os.path.join(self.dataset_dir,
                                                                       f"data/{object_id:02d}/train.txt"))
        else:
            get_split_image_ids_path = lambda object_id: (os.path.join(self.dataset_dir,
                                                                       f"data/{object_id:02d}/test.txt"))

        get_meta_info_path = lambda object_id: (os.path.join(self.dataset_dir, f"data/{object_id:02}/gt.yml"))
        get_model_path = lambda object_id: (os.path.join(self.dataset_dir, f"models/obj_{object_id:02}.ply"))

        for obj_id in self.objects:

            temp_image_ids = pd.read_csv(get_split_image_ids_path(obj_id), header=None).squeeze("columns").tolist()
            for temp_image_id in temp_image_ids:
                self.image_ids.append(temp_image_id)
                self.obj_ids.append(obj_id)

            # load meta info (target rotation, target translation, bounding boxes)
            with open(get_meta_info_path(obj_id), "r") as f:
                all_meta_info = yaml.safe_load(f)
            for image_id in temp_image_ids:
                if obj_id == 2:
                    # the gt.yml file for object id 2 contains info on multiple objects for each image,
                    # so it needs special treatment
                    meta_info = all_meta_info[image_id]  # list of dicts
                    # search for dict with info for object id 2
                    meta_info = next(item for item in meta_info if item["obj_id"] == 2)
                else:
                    # dict containing cam_R_m2c, cam_t_m2c, obj_bb, obj_id
                    meta_info = all_meta_info[image_id][0]
                self.model_to_camera_rotation.append(meta_info["cam_R_m2c"])
                self.model_to_camera_translation.append(meta_info["cam_t_m2c"])  # has unit mm
                x_min, y_min, width, height = meta_info["obj_bb"]
                self.obj_bb.append([x_min, x_min + width, y_min, y_min + height])

            # load object model
            with open(get_model_path(obj_id), "r") as f:
                # skip the first 3 lines
                for _ in range(3):
                    f.readline()
                # number of vertices (= number of model points)
                n_points = int(f.readline().split(" ")[-1])

                # header is 17 rows long, but we only need to skip 13 rows because we already read 4
                # first 3 columns are the x, y and z coordinates of the vertices
                model_points = pd.read_csv(f, header=None, sep=" ", skiprows=13, nrows=n_points,
                                           usecols=[0, 1, 2]).to_numpy()
                self.model_points[obj_id] = model_points / 1000  # divide by 1000 to get meters as unit

            self.x_points = np.array([[i for i in range(self.width)] for j in range(self.height)])
            self.y_points = np.array([[j for i in range(self.width)] for j in range(self.height)])

    def __len__(self):
        return len(self.image_ids)

    def get_rgb_path(self, object_id, image_id):
        # returns path to the rgb image given object id and image id
        # (image id is the index of the image within the folder of objects)
        return os.path.join(self.dataset_dir, f"data/{object_id:02d}/rgb/{image_id:04d}.png")

    def get_depth_path(self, object_id, image_id):
        # returns path to the depth image
        return os.path.join(self.dataset_dir, f"data/{object_id:02d}/depth/{image_id:04d}.png")

    def get_mask_path(self, object_id, image_id):
        if self.mode == "eval":
            return os.path.join(self.dataset_dir, f"segnet_results/{object_id:02d}_label/{image_id:04d}_label.png")
        else:
            return os.path.join(self.dataset_dir, f"data/{object_id:02d}/mask/{image_id:04d}.png")

    def __getitem__(self, idx):
        obj_id = self.obj_ids[idx]
        image_id = self.image_ids[idx]
        img = Image.open(self.get_rgb_path(obj_id, image_id))
        if self.add_noise:
            img = self.trancolor(img)
        depth = np.array(
            Image.open(self.get_depth_path(obj_id, image_id))) / 1000  # divide by 1000 to get meters as unit
        mask = np.array(Image.open(self.get_mask_path(obj_id, image_id)))

        # mask where depth measurement is valid (not equal to 0)
        mask_depth = ma.getmaskarray(ma.masked_not_equal(depth, 0))

        if self.mode == "eval":
            mask_label = ma.getmaskarray(ma.masked_equal(mask, np.array(255)))
        else:
            # mask where mask of object is valid
            mask_label = ma.getmaskarray(ma.masked_equal(mask, np.array([255, 255, 255])))[:, :, 0]

        # combined mask of original mask and depth mask
        mask = mask_label * mask_depth

        # bounding box boundaries
        x_min, x_max, y_min, y_max = self.obj_bb[idx]
        # image cropped to bounding box
        # y corresponds to the row index and x to the column index
        img_cropped = np.array(img)[y_min:y_max, x_min:x_max, :3]
        # indices where the cropped mask is valid
        mask_indices_cropped = mask[y_min:y_max, x_min:x_max].flatten().nonzero()[0]

        # return all zero vector if there is no valid point
        if len(mask_indices_cropped) == 0:
            cc = torch.LongTensor([0])
            return cc, cc, cc, cc, cc, cc
        # downsample points by selecting as many points as needed randomly if there are too many points
        if len(mask_indices_cropped) > self.num_points:
            c_mask = np.zeros(len(mask_indices_cropped), dtype=int)
            c_mask[:self.num_points] = 1
            np.random.shuffle(c_mask)
            mask_indices_cropped = mask_indices_cropped[c_mask.nonzero()]
        # repeat points if there are not enough points
        else:
            mask_indices_cropped = np.pad(mask_indices_cropped, (0, self.num_points - len(mask_indices_cropped)),
                                          'wrap')

        # depth, x and y points at the locations where the object is according to the mask
        depth_masked = depth[y_min:y_max, x_min:x_max].flatten()[mask_indices_cropped][:, np.newaxis].astype(np.float32)
        x_points_masked = self.x_points[y_min:y_max, x_min:x_max].flatten()[mask_indices_cropped][:, np.newaxis].astype(
            np.float32)
        y_points_masked = self.y_points[y_min:y_max, x_min:x_max].flatten()[mask_indices_cropped][:, np.newaxis].astype(
            np.float32)

        # transform image coordinates to camera coordinates
        z_camera_coord = depth_masked
        x_camera_coord = (x_points_masked - self.cx) * z_camera_coord / self.fx
        y_camera_coord = (y_points_masked - self.cy) * z_camera_coord / self.fy
        # object points in camera coordinates
        object_point_cloud = np.concatenate((x_camera_coord, y_camera_coord, z_camera_coord), axis=1)

        model_to_camera_rotation = np.array(self.model_to_camera_rotation[idx]).reshape((3, 3))
        # divide by 1000 to get meters as unit
        model_to_camera_translation = np.array(self.model_to_camera_translation[idx]) / 1000

        if self.add_noise:
            uniform_noise = np.random.uniform(-self.noise_trans, self.noise_trans, (1, 3))
            # add noise to translation vector
            model_to_camera_translation = model_to_camera_translation + uniform_noise
            # add noise to object point cloud
            object_point_cloud = object_point_cloud + uniform_noise + \
                                 np.clip(0.001 * np.random.randn(object_point_cloud.shape[0], 3), -0.005, 0.005)

        # unit vector pointing from each object point to the object center
        point_to_center = model_to_camera_translation - object_point_cloud
        point_to_center = point_to_center / np.linalg.norm(point_to_center, axis=1)[:, None]

        # model points in model coordinates
        model_points = self.model_points[obj_id]
        # downsample model points

        # downsample model points by selecting as many points as needed randomly
        dellist = [j for j in range(0, len(model_points))]
        dellist = random.sample(dellist, len(model_points) - self.num_points_model)
        model_points = np.delete(model_points, dellist, axis=0)

        # model points in camera coordinates
        model_points_camera_coord = np.dot(model_points, model_to_camera_rotation.T)

        return torch.from_numpy(object_point_cloud.astype(np.float32)), \
               torch.LongTensor(np.array([mask_indices_cropped]).astype(np.int32)), \
               self.transform(img_cropped), \
               torch.from_numpy(point_to_center.astype(np.float32)), \
               torch.from_numpy(model_points_camera_coord.astype(np.float32)), \
               torch.from_numpy(model_points.astype(np.float32)), \
               torch.LongTensor([self.objects.index(obj_id)]), \
               torch.from_numpy(model_to_camera_translation.astype(np.float32))

    def get_model_diameter(self, obj_id):
        return self.model_diameter.get(obj_id, None)
