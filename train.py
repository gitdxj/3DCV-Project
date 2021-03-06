import torch
import tensorflow as tf
import time
from torch.autograd import Variable
from loss import Loss
from model.network import PoseNet
from dataset import LinemodDataset

RECORD_AFTER_EVERY = 50  # number of batches after which the loss is recorded


def record_train_metric(writer, train_loss, r_loss, t_loss, reg_loss, step):
    summary = tf.Summary(value=[tf.Summary.Value(tag='train_loss', simple_value=train_loss / RECORD_AFTER_EVERY),
                                tf.Summary.Value(tag='r_loss', simple_value=r_loss / RECORD_AFTER_EVERY),
                                tf.Summary.Value(tag='t_loss', simple_value=t_loss / RECORD_AFTER_EVERY),
                                tf.Summary.Value(tag='reg_loss', simple_value=reg_loss / RECORD_AFTER_EVERY)])
    writer.add_summary(summary, step)


def train_linemod(epochs):
    dataset_path = '/input/Linemod_preprocessed'

    # load training set
    train_set = LinemodDataset(mode='train', dataset_path=dataset_path, cloud_pt_num=500)
    train_loader = torch.utils.data.DataLoader(train_set, batch_size=1, shuffle=True)

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
                cloud = Variable(cloud).cuda()          # shape: 1, 500, 3
                choice = Variable(choice).cuda()        # shape: 1, 1, 500
                img_crop = Variable(img_crop).cuda()    # shape: 1, 3, 80, 80
                target_t = Variable(target_t).cuda()    # shape: 1, 500, 3
                target_r = Variable(target_r).cuda()    # shape: 1, 500, 3
                model_vtx = Variable(model_vtx).cuda()  # shape: 1, 500, 3
                obj_idx = Variable(obj_idx).cuda()      # shape: 1, 1
                gt_t = Variable(gt_t).cuda()            # shape: 1, 3

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


if __name__ == '__main__':
    train_linemod(50)
