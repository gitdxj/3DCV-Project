
import torch
import tensorflow as tf
import numpy as np
import time
from torch.autograd import Variable
# from torch.utils.tensorboard import SummaryWriter

from my_loss import Loss as MY_LOSS
from loss import Loss
from model.network import PoseNet
from dataset import LinemodDataset
from lib.transformations import quaternion_matrix
from model.ransac_voting.ransac_voting_gpu import ransac_voting_layer
from model.knn.__init__ import KNearestNeighbor

RECORD_AFTER_EVERY = 50


def record_train_metric(writer, train_loss, r_loss, t_loss, reg_loss, step):
    summary = tf.Summary(value=[tf.Summary.Value(tag='train_loss', simple_value=train_loss / RECORD_AFTER_EVERY),
                                tf.Summary.Value(tag='r_loss', simple_value=r_loss / RECORD_AFTER_EVERY),
                                tf.Summary.Value(tag='t_loss', simple_value=t_loss / RECORD_AFTER_EVERY),
                                tf.Summary.Value(tag='reg_loss', simple_value=reg_loss / RECORD_AFTER_EVERY)])
    writer.add_summary(summary, step)


def train_linemod(epochs):
    dataset_path = '/input/Linemod_preprocessed'

    # load training set and test set
    train_set = LinemodDataset(mode='train', dataset_path=dataset_path, cloud_pt_num=500)
    test_set = LinemodDataset(mode='test', dataset_path=dataset_path, cloud_pt_num=500)
    train_loader = torch.utils.data.DataLoader(train_set, batch_size=1, shuffle=True)
    test_loader = torch.utils.data.DataLoader(test_set, batch_size=1, shuffle=False)

    # model
    model = PoseNet(cloud_pt_num=500, obj_num=13, rot_num=12)
    model.cuda()

    # loss func
    criterion = MY_LOSS(sym_list=[7, 8], rot_anchors=model.rot_anchors)

    # optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    # record losses on tensorboard
    # writer = SummaryWriter()
    writer = tf.summary.FileWriter('./results')

    current_steps = 0

    start = time.time()

    for epoch in range(epochs):
        # initialize record parameters
        train_loss = 0
        train_r_loss = 0
        train_t_loss = 0
        train_reg_loss = 0

        # set model to training mode
        model.train()
        optimizer.zero_grad()
        for _ in range(10):
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

                obj_diameter = train_set.diameter_dict[train_set.objects[obj_idx]]

                # predict
                pred_r, pred_t, pred_c = model(img_crop, cloud, choice, obj_idx)

                # compute loss
                loss, r_loss, t_loss, reg_loss = criterion(pred_r, pred_t, pred_c, target_r, target_t, model_vtx, obj_idx, obj_diameter)

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
                    # writer.add_scalar('Loss/train', train_loss/RECORD_AFTER_EVERY, current_steps)
                    # writer.add_scalar('Loss/r_loss', r_loss/RECORD_AFTER_EVERY, current_steps)
                    # writer.add_scalar('Loss/t_loss', t_loss/RECORD_AFTER_EVERY, current_steps)
                    # writer.add_scalar('Loss/reg_loss', reg_loss/RECORD_AFTER_EVERY, current_steps)
                    record_train_metric(writer, train_loss, train_r_loss, train_t_loss, train_reg_loss, current_steps)

                    # reset the loss
                    train_loss = 0
                    train_r_loss = 0
                    train_t_loss = 0
                    train_reg_loss = 0
                    print("epoch: ", epoch, "Time:", time.strftime("%Hh %Mm %Ss", time.gmtime(time.time() - start)))
        print("epoch ", epoch, "train finish")
        if epoch % 10 == 0:
            torch.save(model, './posenet.pt')

    torch.save(model, './posenet.pt')




def train_linemod2(epochs):
    dataset_path = '/input/Linemod_preprocessed'

    # load training set and test set
    train_set = LinemodDataset(mode='train', dataset_path=dataset_path, cloud_pt_num=500)
    test_set = LinemodDataset(mode='test', dataset_path=dataset_path, cloud_pt_num=500)
    train_loader = torch.utils.data.DataLoader(train_set, batch_size=1, shuffle=True)
    test_loader = torch.utils.data.DataLoader(test_set, batch_size=1, shuffle=False)

    # objects
    objects = train_set.objects
    diameter_dict = train_set.diameter_dict
    objects_num = len(objects)

    # model
    model = PoseNet(cloud_pt_num=500, obj_num=13, rot_num=12)
    model.cuda()

    # loss func
    criterion = Loss(sym_list=[7, 8], rot_anchors=model.rot_anchors)

    # optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    # record losses on tensorboard
    # writer = SummaryWriter()
    writer = tf.summary.FileWriter('./results')

    current_steps = 0

    knn = KNearestNeighbor(1)

    start = time.time()

    for epoch in range(epochs):
        # initialize record parameters
        train_loss = 0
        train_r_loss = 0
        train_t_loss = 0
        train_reg_loss = 0

        # set model to training mode
        model.train()
        optimizer.zero_grad()
        for _ in range(10):
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

                obj_diameter = diameter_dict[objects[obj_idx]]

                # predict
                pred_r, pred_t, pred_c = model(img_crop, cloud, choice, obj_idx)

                # compute loss
                loss, r_loss, t_loss, reg_loss = criterion(pred_r, pred_t, pred_c, target_r, target_t, model_vtx, obj_idx, obj_diameter)

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
                    # writer.add_scalar('Loss/train', train_loss/RECORD_AFTER_EVERY, current_steps)
                    # writer.add_scalar('Loss/r_loss', r_loss/RECORD_AFTER_EVERY, current_steps)
                    # writer.add_scalar('Loss/t_loss', t_loss/RECORD_AFTER_EVERY, current_steps)
                    # writer.add_scalar('Loss/reg_loss', reg_loss/RECORD_AFTER_EVERY, current_steps)
                    record_train_metric(writer, train_loss, train_r_loss, train_t_loss, train_reg_loss, current_steps)

                    # reset the loss
                    train_loss = 0
                    train_r_loss = 0
                    train_t_loss = 0
                    train_reg_loss = 0
                    print("epoch: ", epoch, "Time:", time.strftime("%Hh %Mm %Ss", time.gmtime(time.time() - start)))

        # evaluation mode
        model.eval()

        test_dis = 0.0
        test_count = 0
        success_count = [0 for i in range(len(test_set.objects))]
        num_count = [0 for i in range(len(test_set.objects))]
        best_test = 0.0

        for j, data in enumerate(test_loader, 0):
            try:
                cloud, choice, img_crop, target_t, target_r, model_vtx, obj_idx, gt_t = data
            except:
                print('No.{0} NOT Pass! Lost detection!'.format(j))
                continue

            cloud = Variable(cloud).cuda()  # shape: 500, 3
            choice = Variable(choice).cuda()  # shape: 1, 500
            img_crop = Variable(img_crop).cuda()  # shape: 3, 80, 80
            target_t = Variable(target_t).cuda()  # shape: 500, 3
            target_r = Variable(target_r).cuda()  # shape: 500, 3
            model_vtx = Variable(model_vtx).cuda()  # shape: 500, 3
            obj_idx = Variable(obj_idx).cuda()  # shape: 1
            gt_t = Variable(gt_t).cuda()  # shape: 3

            obj_diameter = diameter_dict[objects[obj_idx]]

            # predict
            pred_r, pred_t, pred_c = model(img_crop, cloud, choice, obj_idx)

            # compute loss
            loss, r_loss, t_loss, reg_loss = criterion(pred_r, pred_t, pred_c, target_r, target_t, model_vtx, obj_idx, obj_diameter)

            test_count += 1
            # evalaution
            how_min, which_min = torch.min(pred_c, 1)
            pred_r = pred_r[0][which_min[0]].view(-1).cpu().data.numpy()
            pred_r = quaternion_matrix(pred_r)[:3, :3]
            pred_t, pred_mask = ransac_voting_layer(cloud, pred_t)
            pred_t = pred_t.cpu().data.numpy()
            cloud = cloud[0].cpu().detach().numpy()
            pred = np.dot(cloud, pred_r.T) + pred_t
            target = target_r[0].cpu().detach().numpy() + gt_t[0].cpu().data.numpy()
            if obj_idx[0].item() in test_set.get_sym_list():
                pred = torch.from_numpy(pred.astype(np.float32)).cuda().transpose(1, 0).contiguous()
                target = torch.from_numpy(target.astype(np.float32)).cuda().transpose(1, 0).contiguous()
                inds = knn(target.unsqueeze(0), pred.unsqueeze(0))
                target = torch.index_select(target, 1, inds.view(-1) - 1)
                dis = torch.mean(torch.norm((pred.transpose(1, 0) - target.transpose(1, 0)), dim=1), dim=0).item()
            else:
                dis = np.mean(np.linalg.norm(pred - target, axis=1))
            if dis < 0.1 * obj_diameter:
                success_count[obj_idx[0].item()] += 1
            num_count[obj_idx[0].item()] += 1
            test_dis += dis
        # compute accuracy
        accuracy = 0.0
        for i in range(objects_num):
            accuracy += float(success_count[i]) / num_count[i]
        accuracy = accuracy / objects_num
        test_dis = test_dis / test_count
        # tensorboard
        summary = tf.Summary(value=[tf.Summary.Value(tag='accuracy', simple_value=accuracy),
                                    tf.Summary.Value(tag='test_dis', simple_value=test_dis)])
        writer.add_summary(summary, current_steps)
        # save model
        if test_dis < best_test:
            best_test = test_dis
        torch.save(model.state_dict(), './results/pose_model_{1:02d}.pth'.format(epoch))
        print('>>>>>>>>----------epoch {0} test finish---------<<<<<<<<'.format(epoch))







