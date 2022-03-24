import torch
from model.network import PoseNet
from utils import train_linemod2


# img, x, choose, obj = torch.randn(1, 3, 80, 80), torch.randn(1, 500, 3), torch.randint(low=0, high=6400, size=(1, 500)), torch.Tensor([1]).type(torch.int)
# img = img.cuda()
# x = x.cuda()
# choose = choose.cuda()
# obj = obj.cuda()
# model = PoseNet(500, 13, 24)
# model.cuda()
# pred_r, pred_t, pred_c  = model(img, x, choose, obj)
# criterion = Loss(sym_list=[7, 8], rot_anchors=model.rot_anchors)
# = criterion()
# print(loss())

if __name__ == '__main__':
    train_linemod2(50)