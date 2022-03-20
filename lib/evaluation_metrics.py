import torch


def average_distance(prediction, ground_truth):
    """
    Computes the average distance between predicted and ground truth object points
    :param prediction: num_points_model x 3
    :param ground_truth: num_points_model x 3
    :return:
    """
    return torch.mean(torch.norm((prediction - ground_truth), dim=1))


def average_distance_symmetric(prediction, ground_truth):
    """
    For symmetric objects. Computes the average distance from each predicted point
    to the closest ground truth object point
    :param prediction: num_points_model x 3
    :param ground_truth: num_points_model x 3
    :return:
    """
    # pairwise distances between predicted points and ground truth points
    dists = torch.cdist(prediction, ground_truth, p=2)  # num_points_model x num_points_model
    # minimal distances from each predicted point to target points
    min_dists, _ = torch.min(dists, dim=0)  # num_points_model
    mean_dist = torch.mean(min_dists)
    return mean_dist
