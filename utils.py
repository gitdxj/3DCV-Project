from dataset import LinemodDataset
import torch
from torch.autograd import Variable
from torch.utils.tensorboard import SummaryWriter

RECORD_AFTER_EVERY = 100


def train_linemod(epochs):
    dataset_path = '/input/Linemod_preprocessed'

    # load training set and test set
    train_set = LinemodDataset(mode='train', add_noise=True, dataset_dir=dataset_path, num_points=500, noise_trans=0.01)
    test_set = LinemodDataset(mode='test', add_noise=False, dataset_dir=dataset_path, num_points=500)
    train_loader = torch.utils.data.DataLoader(train_set, batch_size=1, shuffle=True)
    test_loader = torch.utils.data.DataLoader(test_set, batch_size=1, shuffle=False)

    # model
    model = ...

    # loss func
    criterion = ...

    # optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    # record losses on tensorboard
    writer = SummaryWriter()

    current_steps = 0

    for epoch in epochs:
        # initialize record parameters
        train_loss = 0
        train_r_loss = 0
        train_t_loss = 0
        train_reg_loss = 0

        # set model to training mode
        model.train()
        optimizer.zero_grad()

        for i, data in enumerate(train_loader, 0):
            cloud, choice, img_crop, target_t, target_r, model_vtx, obj_idx, gt_t = data
            cloud = Variable(cloud).cuda()          # shape: 500, 3
            choice = Variable(choice).cuda()        # shape: 1, 500
            img_crop = Variable(img_crop).cuda()    # shape: 3, 80, 80
            target_t = Variable(target_t).cuda()    # shape: 500, 3
            target_r = Variable(target_r).cuda()    # shape: 500, 3
            model_vtx = Variable(model_vtx).cuda()  # shape: 500, 3
            obj_idx = Variable(obj_idx).cuda()      # shape: 1
            gt_t = Variable(gt_t).cuda()            # shape: 3

            obj_diameter = train_set.get_model_diameter(train_set.objects[obj_idx])

            # predict
            pred_r, pred_t, pred_c = model(img_crop, cloud, choice, obj_idx)

            # compute loss
            loss, r_loss, t_loss, reg_loss = criterion(pred_r, pred_t, pred_c, target_r, target_t, cloud, obj_idx, obj_diameter)

            loss.backward()

            current_steps += 1
            train_loss += loss.item()
            train_r_loss += r_loss.item()
            train_t_loss += t_loss.item()
            train_reg_loss += reg_loss.item()

            if current_steps % RECORD_AFTER_EVERY == 0:
                # backpropagation
                optimizer.step()
                optimizer.zero_grad()

                # show the loss on tensorboard every a certain number of batches
                writer.add_scalar('Loss/train', train_loss/RECORD_AFTER_EVERY, current_steps)
                writer.add_scalar('Loss/r_loss', r_loss/RECORD_AFTER_EVERY, current_steps)
                writer.add_scalar('Loss/t_loss', t_loss/RECORD_AFTER_EVERY, current_steps)
                writer.add_scalar('Loss/reg_loss', reg_loss/RECORD_AFTER_EVERY, current_steps)

                # reset the loss
                train_loss = 0
                train_r_loss = 0
                train_t_loss = 0
                train_reg_loss = 0
        print("epoch ", epoch, "train finish")






