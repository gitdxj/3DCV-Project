import torch
import torch.nn as nn
import torch.nn.functional as F
from psp.pspnet import PSPNet


class PoseNet(nn.Module):
    def __init__(self, cloud_pt_num, obj_num, rot_num, k=16):
        super(PoseNet, self).__init__()
        self.cloud_pt_num = cloud_pt_num
        self.obj_num = obj_num
        self.rot_num = rot_num
        self.k = k

        # TODO: sample rotation

        self.pspnet = PSPNet(sizes=(1, 2, 3, 6), psp_size=512, deep_features_size=256, backend='resnet18')

        # convolution for edge features
        self.edge_conv1 = torch.nn.Conv2d(6, 64, 1)
        self.edge_conv2 = torch.nn.Conv2d(128, 64, 1)
        self.edge_conv3 = torch.nn.Conv2d(128, 128, 1)

        # convolution for psp features
        self.conv1 = torch.nn.Conv1d(32, 64, 1)
        self.conv2 = torch.nn.Conv1d(64, 128, 1)

        self.conv1_t = torch.nn.Conv1d(256, 256, 1)
        self.conv2_t = torch.nn.Conv1d(256, 1024, 1)

        self.conv1_r = torch.nn.Conv1d(256, 256, 1)
        self.conv2_r = torch.nn.Conv1d(256, 1024, 1)



    def forward(self, img_crop, cloud, choice, obj_idx):
        """
        :param img_crop: image window cropped using the bounding box parameter
                         shape: batch_size x 3 x 80 x 80
        :param cloud: cloud points of the objects
                         shape: batch_size x cloud_pt_num x 3
        :param choice: index of cloud points in img_crop
                         shape: batch_size x cloud_pt_num
        :param obj_idx: the index of object in dataset.objects
        :return:
        """
        psp_feat = self.pspnet(img_crop)  # 1 x 32 x 80 x 80

        psp_feat = torch.flatten(psp_feat, start_dim=2)  # 1 x 32 x 80 x 80 --> 1 x 32 x 6400

        psp_feat_channels = psp_feat.shape[1]  # number of channels of psp_feat = 32
        # pixel-wise selection of psp_feat according to cloud points index
        psp_feat = torch.gather(input=psp_feat, dim=2, index=choice.repeat(1, psp_feat_channels, 1))



    def get_nn_idx(self, cloud):
        """
        get k nearest neighbor
        :param cloud: shape batch_size x channel_num x cloud_pt_num
        """
        # get k nearest neighbor
        # calculate squared Euclidean distance between every two points
        cloud_T = torch.transpose(cloud, dim0=1, dim1=2)  # 1 x 500 x 3
        inner = torch.bmm(cloud_T, cloud)  # batch matrix-matrix multiplication: cloud @ cloud_T, shape: 1 x 500 x 500
        square = torch.sum(torch.pow(cloud, 2), dim=1, keepdim=True)  # 1 x 1 x 500

        # distance[i][j]: squared distance between point i and j
        # 1x500x1  + 1x1x500 - 2* 1x500x500  -->  1x500x500
        distance = torch.transpose(square, dim0=1, dim1=2) + square - 2*inner

        # get the index of the topk nearest points, shape: 1 x 500 x k
        _, topk_idx = torch.topk(distance, self.k, largest=False, dim=-1)
        return topk_idx

    def get_edge_feature(self, cloud, topk_idx):
        """
        get edge features from cloud points
        :param cloud: shape: batch_size x channel_num x cloud_pt_num
        :param topk_idx: index of topk nearest points, shape: 1 x 500 x k
        :return: edge_feat: edge features
        """
        batch_size, channel_num, cloud_pt_num = cloud.shape
        # adjust index shape
        # 1 x 500 x k  ->  1 x 1 x 500 x k  ->  1 x c x 500 x k        ->       1 x c x 500*k
        topk_idx = torch.unsqueeze(topk_idx, 1).repeat(1, channel_num, 1, 1).view(batch_size, channel_num, cloud_pt_num*self.k)

        # points selection according to the index, shape: 1 x c x 500 x k
        neighbors = torch.gather(cloud, 2, topk_idx).view(batch_size, channel_num, cloud_pt_num, self.k)

        central = torch.unsqueeze(cloud, 3).repeat(1, 1, 1, self.k)  # shape: 1 x c x 500 x k
        edge_feat = torch.cat((central, neighbors - central), dim=1)  # shape: 1 x 2c x 500 x k
        return edge_feat





