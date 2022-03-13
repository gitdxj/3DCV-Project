from dataset import LinemodDataset
import torch
from torch.autograd import Variable


def train_linemod(epochs):
    dataset_path = '/input/Linemod_preprocessed'

    # load training set and test set
    train_set = LinemodDataset(mode='train', dataset_path=dataset_path, cloud_pt_num=500)
    test_set = LinemodDataset(mode='train', dataset_path=dataset_path, cloud_pt_num=500)
    train_loader = torch.utils.data.DataLoader(train_set, batch_size=1, shuffle=True)
    test_loader = torch.utils.data.DataLoader(test_set, batch_size=1, shuffle=False)

    # model
    model = ...

    # loss func
    criterion = ...

    # optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    for epoch in epochs:
        for i, data in enumerate(train_loader, 0):
            cloud, choice, img_crop, target_t, target_r, model_vtx, obj_idx, gt_t = data
            choice = Variable(choice).cuda()
            img_crop = Variable(img_crop).cuda()
            target_t = Variable(target_t).cuda()
            target_r = Variable(target_r).cuda()
            model_vtx = Variable(model_vtx).cuda()
            obj_idx = Variable(obj_idx).cuda()
            gt_t = Variable(gt_t).cuda()

