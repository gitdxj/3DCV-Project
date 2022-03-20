import torch
import torch.nn as nn
from torch.nn import Conv2d, Conv1d
import torch.nn.functional as F
import numpy as np
from model.psp.pspnet import PSPNet
from lib.transformations import quaternion_from_matrix


class PoseNet(nn.Module):
    def __init__(self, cloud_pt_num, obj_num, rot_num=12, k=16):
        super(PoseNet, self).__init__()
        self.cloud_pt_num = cloud_pt_num
        self.obj_num = obj_num
        self.rot_num = rot_num
        self.k = k

        # TODO: sample rotation
#         self.rot_anchors = torch.Tensor(sample_rotations_12()).cuda()
        self.rot_anchors = sample_rotations_12()

        self.pspnet = PSPNet(sizes=(1, 2, 3, 6), psp_size=512, deep_features_size=256, backend='resnet18', pretrained=False)

        # convolution layer for edge features
        self.edge_conv1 = Conv2d(6, 64, 1)
        self.edge_conv2 = Conv2d(128, 64, 1)
        self.edge_conv3 = Conv2d(128, 128, 1)

        # convolution layer for psp features
        self.conv1 = Conv1d(32, 64, 1)
        self.conv2 = Conv1d(64, 128, 1)

        # translation
        self.conv1_t = Conv1d(256, 256, 1)
        self.conv2_t = Conv1d(256, 1024, 1)
        # translation prediction
        self.conv3_t = Conv1d(1152, 512, 1)
        self.conv4_t = Conv1d(512, 256, 1)
        self.conv5_t = Conv1d(256, 128, 1)
        self.conv6_t = Conv1d(128, self.obj_num * 3, 1)

        # rotation
        self.conv1_r = Conv1d(256, 256, 1)
        self.conv2_r = Conv1d(256, 1024, 1)
        # rotation prediction
        self.conv3_r = torch.nn.Conv1d(1024, 512, 1)
        self.conv4_r = torch.nn.Conv1d(512, 256, 1)
        self.conv5_r = torch.nn.Conv1d(256, 128, 1)
        self.conv6_r = torch.nn.Conv1d(128, self.obj_num * self.rot_num * 4, 1)

        # certainty prediction
        self.conv1_c = Conv1d(1024, 512, 1)
        self.conv2_c = Conv1d(512, 256, 1)
        self.conv3_c = Conv1d(256, 128, 1)
        self.conv4_c = Conv1d(128, self.obj_num * self.rot_num * 1, 1)

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
        batch_size = img_crop.shape[0]

        psp_feat = self.pspnet(img_crop)  # 1 x 32 x 80 x 80

        psp_feat = torch.flatten(psp_feat, start_dim=2)  # 1 x 32 x 80 x 80 --> 1 x 32 x 6400

        psp_feat_channels = psp_feat.shape[1]  # number of channels of psp_feat = 32
        # pixel-wise selection of psp_feat according to cloud points index
        psp_feat = torch.gather(input=psp_feat, dim=2, index=choice.repeat(1, psp_feat_channels, 1))

        x = torch.transpose(cloud, dim0=1, dim1=2)  # shape: 1 x 3 x 500
        y = psp_feat

        # get index of k nearest neighbors for the computation of edge_feature
        topk_idx = self.get_nn_idx(x)

        # cloud branch
        x = F.relu(self.edge_conv1(self.get_edge_feature(x, topk_idx)))  # channel_num: 2*3 -> 64, *2 is from edge_feat
        x, _ = torch.max(x, dim=3, keepdim=False)
        x = F.relu(self.edge_conv2(self.get_edge_feature(x, topk_idx)))  # channel_num: 2*64 -> 64
        x_m, _ = torch.max(x, dim=3, keepdim=False)

        x = F.relu(self.edge_conv3(self.get_edge_feature(x_m, topk_idx)))  # channel_num: 2*64 ->128
        x, _ = torch.max(x, dim=3, keepdim=False)

        # image feature branch
        y_m = F.relu(self.conv1(y))
        y = F.relu(self.conv2(y_m))

        point_feat = torch.cat((x_m, y_m), dim=1)  # 1 x 128 x 500
        fusion = torch.cat((x, y), dim=1)

        # translation
        t_x = F.relu(self.conv1_t(fusion))
        t_x = F.relu(self.conv2_t(t_x))
        t_x = F.adaptive_avg_pool1d(t_x, 1)    # 1 x 1024 x 1

        # concatenation of t_x and point_feat
        t_x = torch.cat((point_feat, t_x.repeat(1, 1, self.cloud_pt_num)), dim=1)

        # rotation
        r_x = F.relu(self.conv1_r(fusion))
        r_x = F.relu(self.conv2_r(r_x))
        r_x = F.adaptive_avg_pool1d(r_x, 1)  # 1 x 1024 x 1

        # translation prediction
        t_x = F.relu(self.conv3_t(t_x))
        t_x = F.relu(self.conv4_t(t_x))
        t_x = F.relu(self.conv5_t(t_x))
        t_x = self.conv6_t(t_x).view(batch_size, self.obj_num, 3, self.cloud_pt_num)

        # certainty prediction
        c_x = F.relu(self.conv1_c(r_x))
        c_x = F.relu(self.conv2_c(c_x))
        c_x = F.relu(self.conv3_c(c_x))
        c_x = torch.sigmoid(self.conv4_c(c_x)).view(batch_size, self.obj_num, self.rot_num)

        # rotation prediction
        r_x = F.relu(self.conv3_r(r_x))
        r_x = F.relu(self.conv4_r(r_x))
        r_x = F.relu(self.conv5_r(r_x))
        r_x = self.conv6_r(r_x).view(batch_size, self.obj_num, self.rot_num, 4)

        # TODO: select prediction

        out_tx = torch.index_select(t_x[0], 0, obj_idx[0])
        out_tx = out_tx.contiguous().transpose(2, 1).contiguous()

        out_cx = torch.index_select(c_x[0], 0, obj_idx[0])  # 1 x rot_num

        out_rx = torch.index_select(r_x[0], 0, obj_idx[0])  # 1 x rot_num x 4
        out_rx = F.normalize(out_rx, p=2, dim=2)  # 1 x rot_num x 4
        rot_anchors = torch.from_numpy(self.rot_anchors).float().cuda()
        rot_anchors = torch.unsqueeze(torch.unsqueeze(rot_anchors, dim=0), dim=3)  # 1 x rot_num x 4 x 1
        out_rx = torch.unsqueeze(out_rx, 2)  # 1 x rot_num x 1 x 4
        out_rx = torch.cat((out_rx[:, :, :, 0], -out_rx[:, :, :, 1], -out_rx[:, :, :, 2], -out_rx[:, :, :, 3], \
                            out_rx[:, :, :, 1], out_rx[:, :, :, 0], out_rx[:, :, :, 3], -out_rx[:, :, :, 2], \
                            out_rx[:, :, :, 2], -out_rx[:, :, :, 3], out_rx[:, :, :, 0], out_rx[:, :, :, 1], \
                            out_rx[:, :, :, 3], out_rx[:, :, :, 2], -out_rx[:, :, :, 1], out_rx[:, :, :, 0], \
                            ), dim=2).contiguous().view(1, self.rot_num, 4, 4)
        out_rx = torch.squeeze(torch.matmul(out_rx, rot_anchors), dim=3)

        return out_rx, out_tx, out_cx

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


def sample_rotations_12():
    group = np.array([[[1, 0, 0], [0, 1, 0], [0, 0, 1]],
                      [[1, 0, 0], [0, -1, 0], [0, 0, -1]],
                      [[-1, 0, 0], [0, 1, 0], [0, 0, -1]],
                      [[-1, 0, 0], [0, -1, 0], [0, 0, 1]],

                      [[0, 1, 0], [0, 0, 1], [1, 0, 0]],
                      [[0, 1, 0], [0, 0, -1], [-1, 0, 0]],
                      [[0, -1, 0], [0, 0, 1], [-1, 0, 0]],
                      [[0, -1, 0], [0, 0, -1], [1, 0, 0]],

                      [[0, 0, 1], [1, 0, 0], [0, 1, 0]],
                      [[0, 0, 1], [-1, 0, 0], [0, -1, 0]],
                      [[0, 0, -1], [1, 0, 0], [0, -1, 0]],
                      [[0, 0, -1], [-1, 0, 0], [0, 1, 0]]])
    quaternion_group = np.zeros((12, 4))
    for i in range(12):
        quaternion_group[i] = quaternion_from_matrix(group[i])
    return quaternion_group.astype(float)



if __name__ == '__main__':
    img, x, choose, obj = torch.randn(1, 3, 80, 80), torch.randn(1, 500, 3), torch.randint(low=0, high=6400, size=(1, 500)), torch.Tensor([1]).type(torch.int)
    model = PoseNet(500, 13, 24)
    model(img, x, choose, obj)
