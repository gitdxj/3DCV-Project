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
    t = nominator_t / denominator_t

    nominator_s = -r1d_r1d * ((r2o - r1o) * r2d).sum(-1) - ((r1o - r2o) * r1d).sum(-1) * r1d_r2d
    denominator_s = denominator_t
    s = nominator_s / denominator_s

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
    p = object_point_coordinates.unsqueeze(0).repeat(num_hypotheses, 1, 1).contiguous()  # num_hypotheses x num_object_points x 3
    v = predicted_unit_vectors.unsqueeze(0).repeat(num_hypotheses, 1, 1).contiguous()  # num_hypotheses x num_object_points x 3

    h_minus_p = h - p

    score = (h_minus_p * v).sum(-1) / \
            ((h_minus_p * h_minus_p).sum(-1) * (v * v).sum(-1)).sqrt()  # num_hypotheses x num_object_points

    inlier_mask = (score >= inlier_threshold)  # num_hypotheses x num_object_points
    inlier_counts = torch.sum(inlier_mask, -1)  # num_hypotheses

    return inlier_mask, inlier_counts


def ransac_voting_layer(cloud, pred_t, round_hyp_num=128, inlier_thresh=0.99, confidence=0.99, max_iter=20, min_num=5,
                        max_num=30000):
    """

    :param cloud: batch_size x num_points x 3, object points
    :param pred_t: batch_size x num_points x 3, predicted unit vectors pointing from each object point to its center
    :param round_hyp_num:
    :param inlier_thresh:
    :param confidence:
    :param max_iter:
    :param min_num:
    :param max_num:
    :return:
    """
    batch_size, num_points, _ = pred_t.size()
    # stores the winning hypothesis point (object center) for each element in the batch
    batch_winning_points = []
    # stores inlier masks for each batch element
    batch_inliers = []

    for batch_index in range(batch_size):

        current_mask = torch.ones(num_points).cuda()

        if num_points < min_num:
            # set winning point to 0
            winning_point = torch.zeros(3, dtype=torch.float32).cuda()
            # no inliers
            inliers = torch.zeros(num_points, dtype=torch.uint8).cuda()
            batch_winning_points.append(winning_point)
            batch_inliers.append(inliers)

        if num_points > max_num:
            selection = torch.zeros(num_points, dtype=torch.float32).cuda().uniform_(0, 1)
            selected_mask = (selection < (max_num / num_points))
            current_mask = current_mask * selected_mask

        num_object_points = torch.sum(current_mask)
        object_point_coordinates = cloud[batch_index].masked_select(current_mask).view(num_object_points, 3)
        predicted_unit_vectors = pred_t[batch_index].masked_select(current_mask).view(num_object_points, 3)

        # counts number of hypothesis
        num_hypothesis = 0
        current_iteration = 0
        # ratio of inliers to object points
        best_winning_ratio = 0
        best_winning_point = torch.zeros(3, dtype=torch.float32).cuda()
        current_winning_ratio = 0

        while (1 - (1 - current_winning_ratio ** 2) ** num_hypothesis) < confidence and current_iteration < max_iter:
            random_indices = torch.randint(low=0, high=num_object_points.item(), size=(round_hyp_num, 2))
            current_hypotheses = generate_hypothesis(predicted_unit_vectors,
                                                     object_point_coordinates, random_indices)  # round_hyp_num x 3
            inlier_mask, inlier_count = count_inliers(predicted_unit_vectors, object_point_coordinates,
                                                      current_hypotheses, inlier_thresh)
            max_inlier_count, winning_hypothesis_index = torch.max(inlier_count)
            winning_hypothesis = current_hypotheses[winning_hypothesis_index]
            current_winning_ratio = max_inlier_count.float() / num_object_points
            if current_winning_ratio > best_winning_ratio:
                best_winning_ratio = current_winning_ratio
                best_winning_point = winning_hypothesis
            num_hypothesis += round_hyp_num
            current_iteration += 1

        # now find point closest to all lines determined by inliers
        # https://math.stackexchange.com/a/1762491
        # todo

    pass
