# code adapted from https://github.com/mentian/object-posenet/blob/master/lib/ransac_voting/ransac_voting_gpu.py
# main differences: indices used to generate hypotheses during RANSAC are resampled in every round
# and the generation of hypotheses and counting of inliers is implemented entirely in pytorch

import torch


def generate_hypothesis(predicted_unit_vectors, object_point_coordinates, indices):
    """
    Computes midpoint of the shortest distance between the two 3D lines defined by
    the pairs of unit vectors and object points. Formula taken from https://math.stackexchange.com/a/2371053
    :param predicted_unit_vectors: num_object_points x 3, predicted unit vectors pointing towards the object center
    :param object_point_coordinates: num_object_points x 3,
    :param indices: num_indices x 2, indices that determine the two unit vectors
    and two object points needed for each 3D line
    :return: midpoint of the shortest distance, num_indices x 3
    """
    selected_unit_vector_pairs = predicted_unit_vectors[indices]  # num_indices x 2 x 3
    r1d = selected_unit_vector_pairs[:, 0, :].contiguous()  # num_indices x 3
    r2d = selected_unit_vector_pairs[:, 1, :].contiguous()  # num_indices x 3
    selected_object_points = object_point_coordinates[indices]  # num_indices x 2 x 3
    r1o = selected_object_points[:, 0, :].contiguous()  # num_indices x 3
    r2o = selected_object_points[:, 1, :].contiguous()  # num_indices x 3

    # precompute dot products, all have shape num_indices
    r1d_r1d = (r1d * r1d).sum(-1)
    r2d_r2d = (r2d * r2d).sum(-1)
    r1d_r2d = (r1d * r2d).sum(-1)

    nominator_t = -((r1o - r2o) * r1d).sum(-1) * r2d_r2d - r1d_r2d * ((r2o - r1o) * r2d).sum(-1)
    denominator_t = r1d_r1d * r2d_r2d - r1d_r2d * r1d_r2d
    t = (nominator_t / denominator_t).unsqueeze(1)  # num_indices x 1

    nominator_s = -r1d_r1d * ((r2o - r1o) * r2d).sum(-1) - ((r1o - r2o) * r1d).sum(-1) * r1d_r2d
    denominator_s = denominator_t
    s = (nominator_s / denominator_s).unsqueeze(1)  # num_indices x 1

    m = 0.5 * (r1o + r1d * t + r2o + r2d * s)  # num_indices x 3
    return m


def count_inliers(predicted_unit_vectors, object_point_coordinates, hypotheses, inlier_threshold):
    """
    Computes the cosine of the angle between the predicted unit vector pointing towards the object center and the
    vector from the corresponding object point to the hypothesised object center. Points where the cosine is greater
    than the inlier threshold are counted as inliers.
    :param predicted_unit_vectors: num_object_points x 3, predicted unit vectors pointing towards the object center
    :param object_point_coordinates: num_object_points x 3
    :param hypotheses: num_hypotheses x 3, hypotheses for the object center
    :param inlier_threshold: threshold for a point to be considered inlier
    :return: mask of inlier points for each hypothesis (num_hypotheses x num_object_points)
    and inlier count of each hypothesis (num_hypotheses)
    """
    # number of object points
    num_object_points, _ = predicted_unit_vectors.size()
    # number of hypotheses
    num_hypotheses, _ = hypotheses.size()

    h = hypotheses.unsqueeze(1).repeat(1, num_object_points, 1).contiguous()  # num_hypotheses x num_object_points x 3
    p = object_point_coordinates.unsqueeze(0).repeat(num_hypotheses, 1,
                                                     1).contiguous()  # num_hypotheses x num_object_points x 3
    v = predicted_unit_vectors.unsqueeze(0).repeat(num_hypotheses, 1,
                                                   1).contiguous()  # num_hypotheses x num_object_points x 3

    h_minus_p = h - p

    score = (h_minus_p * v).sum(-1) / \
            ((h_minus_p * h_minus_p).sum(-1) * (v * v).sum(-1)).sqrt()  # num_hypotheses x num_object_points

    inlier_mask = (score >= inlier_threshold)  # num_hypotheses x num_object_points
    inlier_counts = torch.sum(inlier_mask, -1)  # num_hypotheses

    return inlier_mask, inlier_counts


def point_closest_to_all_lines(point_coordinates, unit_vectors):
    """
    Finds the point closest to all 3D lines. Formula from https://math.stackexchange.com/a/1762491
    :param point_coordinates: num_lines x 3, points on the lines, one per line
    :param unit_vectors: num_lines x 3, unit vectors determining the direction of each line
    :return: point closest to all lines (1 x 3)
    """
    num_lines, _ = point_coordinates.size()
    S = torch.bmm(unit_vectors.view(num_lines, 1, 3).transpose(1, 2),
                  unit_vectors.view(num_lines, 1, 3)) - torch.eye(3)  # num_lines x 3 x 3
    C = torch.bmm(S, point_coordinates.view(num_lines, 3, 1)).sum(0)  # 3 x 1
    S = S.sum(0)  # 3 x 3
    p = torch.matmul(torch.inverse(S), C).permute(1, 0)  # 1 x 3
    return p


def ransac_voting_layer(cloud, pred_t, round_hyp_num=128, inlier_threshold=0.99, confidence=0.99, max_iter=20,
                        min_num=5,
                        max_num=30000):
    """
    Uses RANSAC to determine the object center from the predicted unit vectors pointing towards the object center
    :param cloud: batch_size x num_points x 3, object points
    :param pred_t: batch_size x num_points x 3, predicted unit vectors pointing from each object point to its center
    :param round_hyp_num: number of hypotheses generated per RANSAC round
    :param inlier_threshold: threshold for a point to be considered inlier
    :param confidence: determines condition on when to stop RANSAC iterations
    :param max_iter: maximum number of RANSAC iterations
    :param min_num: minimum number of object points
    :param max_num: maximum number of object points
    :return: object center points in camera coordinates (batch_size x 3)
    and mask of inliers (batch_size x num_points x 1)
    """
    batch_size, num_points, _ = pred_t.size()
    # stores the winning hypothesis point (object center) for each element in the batch
    batch_winning_points = []
    # stores inlier masks for each batch element
    batch_inliers = []

    for batch_index in range(batch_size):
        # mask of all object points
        current_mask = torch.ones((num_points, 1), dtype=torch.bool, device=pred_t.device)

        if num_points < min_num:
            # set winning point to 0
            winning_point = torch.zeros((1, 3), dtype=torch.float32, device=pred_t.device)
            # no inliers
            inliers = torch.zeros((num_points, 1), dtype=torch.uint8, device=pred_t.device)
            batch_winning_points.append(winning_point)
            batch_inliers.append(inliers)
            continue

        if num_points > max_num:
            # downsample randomly
            selection = torch.zeros((num_points, 1), dtype=torch.float32, device=pred_t.device).uniform_(0, 1).uniform_(0, 1)
            selected_mask = (selection < (max_num / num_points))
            current_mask = current_mask * selected_mask  # mask for points selected by downsampling

        # number of object points considered (less than num_points if points were downsampled)
        num_object_points = torch.sum(current_mask)
        # object points of this batch element
        object_point_coordinates = cloud[batch_index].masked_select(current_mask).view(num_object_points, 3)
        # predicted unit vectors of this batch element
        predicted_unit_vectors = pred_t[batch_index].masked_select(current_mask).view(num_object_points, 3)

        # counts number of hypothesis
        num_hypothesis = 0
        # counts RANSAC iterations
        current_iteration = 0
        # best ratio of inliers to object points
        best_winning_ratio = 0
        # highest inlier count
        best_inlier_count = 0
        # inlier mask for hypothesis with highest inlier count
        best_winning_inlier_mask = torch.zeros((num_object_points.item(), 1), dtype=torch.bool, device=pred_t.device)
        # best ratio of inliers to object points for all hypothesis from the current RANSAC round
        current_winning_ratio = 0

        while (1 - (1 - current_winning_ratio ** 2) ** num_hypothesis) < confidence and current_iteration < max_iter:
            # randomly sample round_hyp_num pairs of indices that determine the pairs of predicted unit vectors
            # from which a hypothesis is generated
            random_indices = torch.randint(low=0, high=num_object_points.item(), size=(round_hyp_num, 2))
            # hypotheses for the object center
            current_hypotheses = generate_hypothesis(predicted_unit_vectors,
                                                     object_point_coordinates, random_indices)  # round_hyp_num x 3
            # mask of inliers and inlier count for each hypothesis
            inlier_mask, inlier_count = count_inliers(predicted_unit_vectors, object_point_coordinates,
                                                      current_hypotheses,
                                                      inlier_threshold)  # num_hypotheses x num_object_points and num_hypotheses
            # find hypothesis with most inliers
            max_inlier_count, winning_hypothesis_index = torch.max(inlier_count, dim=0)
            current_winning_ratio = max_inlier_count / num_object_points
            if current_winning_ratio > best_winning_ratio:
                best_winning_ratio = current_winning_ratio
                best_winning_inlier_mask = inlier_mask[winning_hypothesis_index].unsqueeze(1)  # num_object_points x 1
                best_inlier_count = max_inlier_count
            num_hypothesis += round_hyp_num
            current_iteration += 1

        predicted_unit_vectors_inlier = predicted_unit_vectors.masked_select(best_winning_inlier_mask).view(
            best_inlier_count, 3)  # num_inliers x 3
        # make sure unit vectors are normalized
        predicted_unit_vectors_inlier = predicted_unit_vectors_inlier / torch.linalg.norm(predicted_unit_vectors_inlier,
                                                                                          dim=1,
                                                                                          keepdim=True)  # num_inliers x 3
        object_point_coordinates_inliers = object_point_coordinates.masked_select(best_winning_inlier_mask).view(
            best_inlier_count, 3)  # num_inliers x 3

        # now find point closest to all lines determined by inliers
        p = point_closest_to_all_lines(object_point_coordinates_inliers, predicted_unit_vectors_inlier)  # 1 x 3
        batch_winning_points.append(p)

        # determine mask of inliers for all object points (this is necessary in case the object points were downsampled)
        # indices of the mask of the object points used for RANSAC
        current_mask_indices = current_mask.squeeze(1).nonzero().view(num_object_points, 1)  # num_object_points x 1
        inlier_mask_all_points = current_mask  # num_points x 1
        inlier_mask_all_points.scatter_(0, current_mask_indices, best_winning_inlier_mask)  # num_points x 1
        batch_inliers.append(inlier_mask_all_points.unsqueeze(0))

    batch_winning_points = torch.cat(batch_winning_points)  # batch_size x 3
    batch_inliers = torch.cat(batch_inliers)  # batch_size x num_points x 1
    return batch_winning_points, batch_inliers
