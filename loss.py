import torch
import torch.nn.functional as F
from torch.nn.modules.loss import _Loss


def loss_calculation(pred_r, pred_t, pred_c, target_r, target_t, model_points, idx, obj_diameter, rot_anchors, sym_list):
    """

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
    :param sym_list: todo indices or obj ids ?
    :return:
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
    # clone diagonal elements so they don't get set to zero in the next step
    diagonal_elements = torch.diagonal(inner_product, dim1=1, dim2=2).clone()  # batch_size x num_rotations
    # set diagonal of inner product to zero because we don't want to consider them when taking the maximum
    # this step is omitted in the code from https://github.com/mentian/object-posenet/blob/master/lib/loss.py
    # (maybe a bug?)
    torch.diagonal(inner_product, dim1=1, dim2=2).zero_()
    loss_reg = F.threshold((torch.max(inner_product, 2)[0] - diagonal_elements), 0.001, 0)
    loss_reg = torch.mean(loss_reg)

    # rotation loss

    # translation loss
    loss_t = F.smooth_l1_loss(pred_t, target_t, reduction='mean')

    # total loss

    pass


class Loss(_Loss):
    def __init__(self, sym_list, rot_anchors):
        super(Loss, self).__init__(True)
        self.sym_list = sym_list
        self.rot_anchors = rot_anchors

    def forward(self, pred_r, pred_t, pred_c, target_r, target_t, model_points, idx, obj_diameter):
        """
        """
        return loss_calculation(pred_r, pred_t, pred_c, target_r, target_t, model_points, idx, obj_diameter, self.rot_anchors, self.sym_list)
