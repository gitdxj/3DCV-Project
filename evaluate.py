import torch
from torch.autograd import Variable
import numpy as np

from lib.evaluation_metrics import average_distance, average_distance_symmetric
from lib.transformations import quaternion_matrix
from model.network import PoseNet
from dataset import LinemodDataset
from model.ransac_voting.ransac_voting import ransac_voting_layer


dataset_path = '/input/Linemod_preprocessed'
path_to_trained_model = "./posenet.pt"
result_path = 'results/results.txt'

dataset = LinemodDataset(mode='eval', dataset_path=dataset_path, cloud_pt_num=500)
test_loader = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=False)

object_list = dataset.objects
symmetric_object_indices = dataset.get_sym_list()

# counts how often the object pose was successfully estimated for each object
object_success_count = [0 for _ in range(len(object_list))]
# counts the number of times each object has been evaluated
evaluated_objects = [0 for _ in range(len(object_list))]

error_data = 0

model = torch.load(path_to_trained_model)
model.cuda()
model.eval()
# model = PoseNet(cloud_pt_num=500, obj_num=len(object_list))
# model.cuda()
# model.load_state_dict(torch.load(path_to_trained_model))
# model.eval()



for i, data in enumerate(test_loader, 0):
    try:
        cloud, choice, img_crop, target_t, target_r, model_vtx, obj_idx, gt_t = data
    except:
        error_data += 1
        print('No.{0} NOT Pass! Lost detection!'.format(i))
        continue

    # # output is None if the mask from the output of the SegNet is empty because the object couldn't be detected
    # if cloud is None or choice is None or img_crop is None or target_t is None \
    #         or target_r is None or model_vtx is None or obj_idx is None or gt_t is None:
    #     continue

    cloud = Variable(cloud).cuda()  # shape: 500, 3
    choice = Variable(choice).cuda()  # shape: 1, 500
    img_crop = Variable(img_crop).cuda()  # shape: 3, 80, 80
    target_t = Variable(target_t).cuda()  # shape: 500, 3
    target_r = Variable(target_r).cuda()  # shape: 500, 3
    model_vtx = Variable(model_vtx).cuda()  # shape: 500, 3
    obj_idx = Variable(obj_idx).cuda()  # shape: 1
    gt_t = Variable(gt_t).cuda()  # shape: 3

    obj_diameter = dataset.get_obj_diameter(object_list[obj_idx.item()])

    # prediction
    pred_r, pred_t, pred_c = model(img_crop, cloud, choice, obj_idx)
    pred_t, pred_mask = ransac_voting_layer(cloud.cpu(), pred_t.cpu())
    pred_t = pred_t.cpu().data.numpy()
    how_min, which_min = torch.min(pred_c, 1)
    pred_r = pred_r[0][which_min[0]].view(-1).cpu().data.numpy()
    pred_r = quaternion_matrix(pred_r)[:3, :3]
    model_vtx = model_vtx[0].cpu().detach().numpy()
    pred = np.dot(model_vtx, pred_r.T) + pred_t

    # ground truth object points in camera coordinates
    target = target_r[0].cpu().detach().numpy() + gt_t.cpu().data.numpy()[0]

#     print("target:", target.shape)  # 500 x 3
#     print("pred:", pred.shape)      # 500 x 3

    if obj_idx.item() in symmetric_object_indices:
        distance = average_distance_symmetric(torch.Tensor(pred), torch.Tensor(target))
    else:
        distance = average_distance(torch.Tensor(pred), torch.Tensor(target))

    if distance < 0.1 * obj_diameter:
        object_success_count[obj_idx.item()] += 1
    print("obj:", object_list[obj_idx.item()], "distance:", distance, "diameter:", obj_diameter)

    evaluated_objects[obj_idx.item()] += 1


with open(result_path, "w") as f:
    for idx, (success_count, evaluation_num) in enumerate(zip(object_success_count, evaluated_objects)):
        f.write(f"accuracy for object {object_list[idx]}: {success_count / evaluation_num} \n")
    f.write(f"overall accuracy: {sum(object_success_count) / sum(evaluated_objects)} \n")
