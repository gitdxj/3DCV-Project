# code adapted from https://github.com/mentian/object-posenet/blob/master/lib/loss.py
# main differences: this version supports batch sizes greater than 1
# and has a slightly different computation of the regularization loss & the rotation loss

import torch
import torch.nn.functional as F
from torch.nn.modules.loss import _Loss


def loss_calculation(pred_r, pred_t, pred_c, target_r, target_t, model_points, idx, obj_diameter, rot_anchors,
                     sym_list):
    """
    Computes the loss function, which consists of regularization loss, rotation loss and translation loss
    :param pred_r: batch_size x num_rotations x 4,
    predicted rotations (obtained by multiplying predicted deviation from rotation anchor with rotation anchor), quaternions
    :param pred_t: batch_size x num_points x 3, predicted unit vector from each object point to object center
    :param pred_c: batch_size x num_rotations, predicted uncertainty score for each predicted deviation from the rotation anchor
    :param target_r: batch_size x num_points_model x 3, model points with ground truth rotation (model to camera coord) applied
    :param target_t: batch_size x num_points x 3, unit vectors from each object point to ground truth object center
    :param model_points: batch_size x num_points_model x 3, model points in model coordinates
    :param idx: batch_size x 1, index of the object in the list of objects
    :param obj_diameter: batch_size x 1
    :param rot_anchors: num_rotations x 4, rotation anchors as quaternions
    :param sym_list: indices of the symmetric objects in the list of objects
    :return: total loss, rotation loss, translation loss, regularization loss
    """
    # batch size and number of object points
    batch_size, num_points, _ = pred_t.size()
    # number of rotation anchors
    _, num_rotations, _ = pred_r.size()
    # number of model points
    _, num_points_model, _ = model_points.size()

    # regularization loss
    rot_anchors = torch.from_numpy(rot_anchors).float().cuda()
    rot_anchors = rot_anchors.unsqueeze(0).repeat(batch_size, 1, 1).permute(0, 2, 1)  # batch_size x 4 x num_rotations
    # inner product of each predicted rotation with each rotation anchor
    inner_product = torch.bmm(pred_r, rot_anchors)  # batch_size x num_rotations x num_rotations
    # clone diagonal elements, so they don't get set to zero in the next step
    diagonal_elements = torch.diagonal(inner_product, dim1=1, dim2=2).clone()  # batch_size x num_rotations
    # set diagonal of inner product to zero because we don't want to consider them when taking the maximum
    # this step is omitted in the code from https://github.com/mentian/object-posenet/blob/master/lib/loss.py
    # (maybe a bug?)
    torch.diagonal(inner_product, dim1=1, dim2=2).zero_()
    loss_reg = F.threshold((torch.max(inner_product, 2)[0] - diagonal_elements), 0.001, 0)
    loss_reg = torch.mean(loss_reg)

    # rotation loss

    # convert the quaternion representation to a matrix representation and take the transpose:
    # the 3x3 matrix representing the rotation of a quaternion (qw, qx, qy, qz) is
    # [[1 - 2*qy**2 - 2*qz**2, 2*qx*qy - 2*qz*qw,     2*qx*qz + 2*qy*qw],
    #  [2*qx*qy + 2*qz*qw,     1 - 2*qx**2 - 2*qz**2, 2*qy*qz - 2*qx*qw],
    #  [2*qx*qz - 2*qy*qw,     2*qy*qz + 2*qx*qw,     1 - 2*qx**2 - 2*qy**2]]
    # the transpose of this matrix is thus
    # [[1 - 2*qy**2 - 2*qz**2, 2*qx*qy + 2*qz*qw,     2*qx*qz - 2*qy*qw],
    #  [2*qx*qy - 2*qz*qw,     1 - 2*qx**2 - 2*qz**2, 2*qy*qz + 2*qx*qw],
    #  [2*qx*qz + 2*qy*qw,     2*qy*qz - 2*qx*qw,     1 - 2*qx**2 - 2*qy**2]]
    qw = pred_r[:, :, 0]
    qx = pred_r[:, :, 1]
    qy = pred_r[:, :, 2]
    qz = pred_r[:, :, 3]
    pred_rotation_matrix_transposed = torch.cat(
        ((1.0 - 2.0 * (qy ** 2 + qz ** 2)).view(batch_size, num_rotations, 1),
         (2.0 * qx * qy + 2.0 * qz * qw).view(batch_size, num_rotations, 1),
         (2.0 * qx * qz - 2.0 * qw * qy).view(batch_size, num_rotations, 1),
         (2.0 * qx * qy - 2.0 * qw * qz).view(batch_size, num_rotations, 1),
         (1.0 - 2.0 * (qx ** 2 + qz ** 2)).view(batch_size, num_rotations, 1),
         (2.0 * qy * qz + 2.0 * qw * qx).view(batch_size, num_rotations, 1),
         (2.0 * qx * qz + 2.0 * qw * qy).view(batch_size, num_rotations, 1),
         (2.0 * qy * qz - 2.0 * qw * qx).view(batch_size, num_rotations, 1),
         (1.0 - 2.0 * (qx ** 2 + qy ** 2)).view(batch_size, num_rotations, 1)
         ), dim=2).contiguous().view(batch_size * num_rotations, 3, 3)

    model_points = model_points.view(batch_size, 1, num_points_model, 3)\
        .repeat(1, num_rotations, 1, 1).view(batch_size * num_rotations, num_points_model, 3)

    # model points multiplied with the predicted rotation from model to camera coordinates
    pred_r = torch.bmm(model_points,
                       pred_rotation_matrix_transposed)  # batch_size * num_rotations x num_points_model x 3

    # model points multiplied with ground truth rotation from model to camera coordinates
    target_r = target_r.unsqueeze(1).repeat(1, num_rotations, 1, 1).contiguous().view(batch_size * num_rotations,
                                                                                      num_points_model, 3)

    shape_match_loss = torch.empty((batch_size, num_rotations)).cuda()  # stores the ShapeMatch loss for each predicted rotation

    # compute the ShapeMatch loss for each element of the batch
    for batch_idx, obj_idx in enumerate(idx):
        current_pred = pred_r.view(batch_size, num_rotations, num_points_model, 3)[
            batch_idx]  # num_rotations x num_points_model x 3
        current_target = target_r.view(batch_size, num_rotations, num_points_model, 3)[
            batch_idx]  # num_rotations x num_points_model x 3
        if obj_idx.item() in sym_list:  # case that the object is symmetric
            # pairwise distances between predicted points and ground truth points
            dists = torch.cdist(current_pred, current_target,
                                p=2)  # num_rotations x num_points_model x num_points_model
            # minimal distances from each predicted point to target points
            min_dists, _ = torch.min(dists, dim=1)  # num_rotations x num_points_model
            mean_dist = torch.mean(min_dists, dim=1)  # num_rotations
            shape_match_loss[batch_idx] = mean_dist
        else:
            shape_match_loss[batch_idx] = torch.mean(torch.norm((current_pred - current_target), dim=2), dim=1)  # num_rotations

    dis = shape_match_loss / obj_diameter  # normalize by object diameter
    pred_c = pred_c.contiguous().view(batch_size * num_rotations)
    loss_r = torch.mean(dis / pred_c + torch.log(pred_c))

    # translation loss
    loss_t = F.smooth_l1_loss(pred_t, target_t, reduction='mean')

    # total loss
    loss = loss_r + 2.0 * loss_reg + 5.0 * loss_t
    return loss, loss_r, loss_t, loss_reg


class Loss(_Loss):
    def __init__(self, sym_list, rot_anchors):
        """
        :param sym_list: indices of the symmetric objects in the list of objects
        :param rot_anchors: num_rotations x 4, rotation anchors as quaternions
        """
        super(Loss, self).__init__(True)
        self.sym_list = sym_list
        self.rot_anchors = rot_anchors

    def forward(self, pred_r, pred_t, pred_c, target_r, target_t, model_points, idx, obj_diameter):
        return loss_calculation(pred_r, pred_t, pred_c, target_r, target_t, model_points, idx, obj_diameter, self.rot_anchors, self.sym_list)
