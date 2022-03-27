import torch
from torch.autograd import Variable
import numpy as np

from utils.evaluation_metrics import average_distance, average_distance_symmetric
from dataset import LinemodDataset
from model.network import get_prediction_from_model_output


dataset_path = '/input/Linemod_preprocessed'
path_to_trained_model = "./posenet.pt"
result_path = 'results/results.txt'

dataset = LinemodDataset(mode='eval', dataset_path=dataset_path, cloud_pt_num=500)
test_loader = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=False)

object_list = dataset.objects  # list of object ids
symmetric_object_indices = dataset.get_sym_list()  # indices of symmetric objects in object list

# counts how often the object pose was successfully estimated for each object
object_success_count = [0 for _ in range(len(object_list))]
# counts the number of times each object has been evaluated
evaluated_objects = [0 for _ in range(len(object_list))]

model = torch.load(path_to_trained_model)
model.cuda()

for i, data in enumerate(test_loader, 0):
    try:
        cloud, choice, img_crop, target_t, target_r, model_vtx, obj_idx, gt_t = data
    except:
        # error is raised if the mask from the output of the SegNet is empty because the object couldn't be detected
        print(f"Skipping index {i} because no object was detected.")
        continue

    cloud = Variable(cloud).cuda()  # shape: 1, 500, 3
    choice = Variable(choice).cuda()  # shape: 1, 1, 500
    img_crop = Variable(img_crop).cuda()  # shape: 1, 3, 80, 80
    target_t = Variable(target_t).cuda()  # shape: 1, 500, 3
    target_r = Variable(target_r).cuda()  # shape: 1, 500, 3
    model_vtx = Variable(model_vtx).cuda()  # shape: 1, 500, 3
    obj_idx = Variable(obj_idx).cuda()  # shape: 1, 1
    gt_t = Variable(gt_t).cuda()  # shape: 1, 3

    obj_diameter = dataset.get_obj_diameter(object_list[obj_idx.item()])

    # prediction
    pred_r, pred_t, pred_c = model(img_crop, cloud, choice, obj_idx)
    pred_r, pred_t = get_prediction_from_model_output(pred_r, pred_t, pred_c, cloud)

    model_vtx = model_vtx[0].cpu().detach().numpy()
    # object points with predicted transformation to camera coordinates applied
    pred = np.dot(model_vtx, pred_r.T) + pred_t

    # ground truth object points in camera coordinates
    target = target_r[0].cpu().detach().numpy() + gt_t.cpu().data.numpy()[0]

    if obj_idx.item() in symmetric_object_indices:
        # ADD-S metric
        distance = average_distance_symmetric(torch.Tensor(pred), torch.Tensor(target))
    else:
        # ADD metric
        distance = average_distance(torch.Tensor(pred), torch.Tensor(target))

    if distance < 0.1 * obj_diameter:
        object_success_count[obj_idx.item()] += 1
    print("obj:", object_list[obj_idx.item()], "distance:", distance, "diameter:", obj_diameter)

    evaluated_objects[obj_idx.item()] += 1


with open(result_path, "w") as f:
    for idx, (success_count, evaluation_num) in enumerate(zip(object_success_count, evaluated_objects)):
        f.write(f"accuracy for object {object_list[idx]}: {success_count / evaluation_num} \n")
    f.write(f"overall accuracy: {sum(object_success_count) / sum(evaluated_objects)} \n")
