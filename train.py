import argparse

from torch import optim
from torch.utils.data.dataloader import DataLoader

from metrics import *
from model import *
from utils import *
import pickle
import torch
import random

import sys
import os
import time

from torch.utils.tensorboard import SummaryWriter
import os
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"
writer = SummaryWriter("runs_10_10")

parser = argparse.ArgumentParser()

parser.add_argument('--obs_len', type=int, default=10)
parser.add_argument('--pred_len', type=int, default=10)
#parser.add_argument('--dataset', default='eth',
                  #  help='eth,hotel,univ,zara1,zara2')
parser.add_argument('--dataset', default='eth',
                    help='eth,hotel,univ,zara1,zara2')
# Training specifc parameters
parser.add_argument('--batch_size', type=int, default=32,
                    help='minibatch size')
parser.add_argument('--num_epochs', type=int, default=200,
                    help='number of epochs')
parser.add_argument('--clip_grad', type=float, default=None,
                    help='gadient clipping')
parser.add_argument('--lr', type=float, default=0.00001,
                    help='learning rate')
# parser.add_argument('--momentum', type=float, default=0.9,
#                     help='momentum of lr')
# parser.add_argument('--weight_decay', type=float, default=0.0001,
#                     help='weight_decay on l2 reg')
# parser.add_argument('--lr_sh_rate', type=int, default=100,
#                     help='number of steps to drop the lr')
parser.add_argument('--milestones', type=int, default=[0, 100],
                    help='number of steps to drop the lr')
parser.add_argument('--use_lrschd', action="store_true", default=False,
                    help='Use lr rate scheduler')
parser.add_argument('--tag', default='AddGCN_10_10', help='personal tag for the model ')
parser.add_argument('--gpu_num', default="0", type=str)

args = parser.parse_args()


print("Training initiating....")
print(args)

# 控制台输出记录到文件
class Logger(object):
    def __init__(self, file_name="Default.log", stream=sys.stdout):
        self.terminal = stream
        self.log = open(file_name, "a")

    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)

    def flush(self):
        pass



def graph_loss(V_pred, V_target):
    return bivariate_loss(V_pred, V_target)


metrics = {'train_loss': [], 'val_loss': []}
constant_metrics = {'min_val_epoch': -1, 'min_val_loss': 9999999999999999, 'min_train_epoch': -1,
                    'min_train_loss': 9999999999999999}


def train(epoch, model, optimizer, checkpoint_dir, loader_train):
    global metrics, constant_metrics
    model.train()
    loss_batch = 0
    batch_count = 0
    is_fst_loss = True
    loader_len = len(loader_train)
    turn_point = int(loader_len / args.batch_size) * args.batch_size + loader_len % args.batch_size - 1

    for cnt, batch in enumerate(loader_train):
        batch_count += 1

        # Get data 获取数据
        batch = [tensor.cuda() for tensor in batch]
        obs_traj, pred_traj_gt, obs_traj_rel, pred_traj_gt_rel, non_linear_ped, \
        loss_mask, V_obs, V_tr = batch

        # print("obs_len---",obs_traj.shape)[1, 87, 4, 10])

        # obs_traj observed absolute coordinate [1 N 2 obs_len]    N是行人数
        # pred_traj_gt ground truth absolute coordinate [1 N 2 pred_len]
        # obs_traj_rel velocity of observed trajectory [1 N 2 obs_len]
        # pred_traj_gt_rel velocity of ground-truth [1 N 2 pred_len]
        # non_linear_ped 0/1 tensor indicated whether the trajectory of pedestrians n is linear [1 N]
        # loss_mask 0/1 tensor indicated whether the trajectory point at time t is loss [1 N obs_len+pred_len]
        # V_obs input graph of observed trajectory represented by velocity  [1 obs_len N 3]   #所以这里写的完全正确，【obs_len,行人数N，3】 1表示batch_size ,128
        # V_tr target graph of ground-truth represented by velocity  [1 pred_len N 2]
        # V_tr = V_tr[:,0:2]
        # print("V-TR---",V_tr.shape)

        identity_spatial = torch.ones((V_obs.shape[1], V_obs.shape[2], V_obs.shape[2]), device='cuda') * \
                           torch.eye(V_obs.shape[2], device='cuda')  # [obs_len N N]  -->  [8,N,N]     ----[obs_len,N,N] 每个obs_len里面 的N*N都是对角为1，其余为0 的矩阵
                                                                     #torch.eye(V_obs.shape[2]）--->torch.Size([57, 57])
        identity_temporal = torch.ones((V_obs.shape[2], V_obs.shape[1], V_obs.shape[1]), device='cuda') * \
                            torch.eye(V_obs.shape[1], device='cuda')  # [N obs_len obs_len] -->  [N,8,8]  -----torch.Size([16, 8, 8])
        identity = [identity_spatial, identity_temporal]     #这个就是将两个矩阵放在一起了，identity[0]=identity_spatial,-----

        optimizer.zero_grad()

#V_obs:[128,8,57,3]    空间邻接矩阵identity_spatial：[8,57,57]---  ---时间邻接矩阵identity_temporal：[57,8,8]
        V_pred = model(V_obs, identity)  # A_obs <8, #, #>    #V是2个 8*57的矩阵,,A是8个57*57的矩阵     V_obs是[1 obs_len N 3]   空间iden：（8，N，N）,时间就是（N，8，8）
        V_pred = V_pred.squeeze()     #[128,12,57,5] 最后为什么是5--就是5 高斯分布的参数
        V_tr = V_tr.squeeze()      #[128,12,57,2]删掉一维度之后 变成 三维  应该是 变成 预测长度、行人数、坐标

        if batch_count % args.batch_size != 0 and cnt != turn_point:
            l = graph_loss(V_pred, V_tr)
            if is_fst_loss:   #train方法最开始设定的值是true
                loss = l
                is_fst_loss = False
            else:
                loss += l

        else:
            loss = loss / args.batch_size
            is_fst_loss = True
            loss.backward()

            if args.clip_grad is not None:     #默认是10
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip_grad)

            optimizer.step()
            # Metrics
            loss_batch += loss.item()
            print('TRAIN:', '\t Epoch:', epoch, '\t Loss:', loss_batch / batch_count) #每个batch都要打印一次
    metrics['train_loss'].append(loss_batch / batch_count)

    if metrics['train_loss'][-1] < constant_metrics['min_train_loss']:    #如果train_loss中的最后一个数值 <  最小训练loss
        constant_metrics['min_train_loss'] = metrics['train_loss'][-1]    #就将他复制给最小训练loss  -----这就是将最小训练loss保存
        constant_metrics['min_train_epoch'] = epoch
        torch.save(model.state_dict(), checkpoint_dir + 'train_best.pth')  # OK


def vald(epoch, model, checkpoint_dir, loader_val):
    global metrics, constant_metrics
    model.eval()
    loss_batch = 0
    batch_count = 0
    is_fst_loss = True
    loader_len = len(loader_val)
    turn_point = int(loader_len / args.batch_size) * args.batch_size + loader_len % args.batch_size - 1

    for cnt, batch in enumerate(loader_val):
        batch_count += 1
        # Get data
        batch = [tensor.cuda() for tensor in batch]
        obs_traj, pred_traj_gt, obs_traj_rel, pred_traj_gt_rel, non_linear_ped, \
        loss_mask, V_obs, V_tr = batch

        with torch.no_grad():
            identity_spatial = torch.ones((V_obs.shape[1], V_obs.shape[2], V_obs.shape[2])) * torch.eye(
                V_obs.shape[2])
            identity_temporal = torch.ones((V_obs.shape[2], V_obs.shape[1], V_obs.shape[1])) * torch.eye(
                V_obs.shape[1])
            identity_spatial = identity_spatial.cuda()
            identity_temporal = identity_temporal.cuda()
            identity = [identity_spatial, identity_temporal]

            V_pred = model(V_obs, identity)  # A_obs <8, #, #>

            V_pred = V_pred.squeeze()
            V_tr = V_tr.squeeze()

            if batch_count % args.batch_size != 0 and cnt != turn_point:
                l = graph_loss(V_pred, V_tr)

                if is_fst_loss:
                    loss = l
                    is_fst_loss = False
                else:
                    loss += l

            else:
                loss = loss / args.batch_size
                is_fst_loss = True
                # Metrics
                loss_batch += loss.item()
                print('VALD:', '\t Epoch:', epoch, '\t Loss:', loss_batch / batch_count)
    metrics['val_loss'].append(loss_batch / batch_count)

    if metrics['val_loss'][-1] < constant_metrics['min_val_loss']:
        constant_metrics['min_val_loss'] = metrics['val_loss'][-1]
        constant_metrics['min_val_epoch'] = epoch
        torch.save(model.state_dict(), checkpoint_dir + 'val_best.pth')  # OK


def main(args):
    obs_seq_len = args.obs_len  #观察的长度，设为8  10
    pred_seq_len = args.pred_len  #预测的长度，设为12 30

    data_set = './dataset/' + args.dataset + '/'
    #创建数据实例，从哪里读取数据，和数据的处理
    dset_train = TrajectoryDataset(
        data_set + 'train/',
        obs_len=obs_seq_len,
        pred_len=pred_seq_len,
        skip=1)

    loader_train = DataLoader(
        dset_train,
        batch_size=1,  # This is irrelative to the args batch size parameter    每个batch load多少样本  默认是1
        shuffle=True,
        num_workers=0)      #多进程，默认为0  如果出现  brokenerror这个问题的时候，就要检测workers是不是为0

    dset_val = TrajectoryDataset(
        data_set + 'val/',
        obs_len=obs_seq_len,
        pred_len=pred_seq_len,
        skip=1)

    loader_val = DataLoader(
        dset_val,
        batch_size=1,  # This is irrelative to the args batch size parameter
        shuffle=False,
        num_workers=0)

    print('Training started ...')
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_num

    model = TrajectoryModel(number_asymmetric_conv_layer=2, embedding_dims=64, number_gcn_layers=1, dropout=0,
                            obs_len=args.obs_len, pred_len=args.pred_len, out_dims=5).cuda()

    # optimizer = optim.Adam(model.parameters(), lr=0.01, weight_decay=0.0001)
    optimizer = optim.Adam(model.parameters(), lr=args.lr)

    if args.use_lrschd:   #默认是true
        scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=[0, 100], gamma=0.1)

    # if args.use_lrschd:  # 默认是false
    #     scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=args.lr_sh_rate, gamma=0.2)

    checkpoint_dir = './checkpoints/' + args.tag + '/' + args.dataset + '/'

    if not os.path.exists(checkpoint_dir):
        os.makedirs(checkpoint_dir)

    with open(checkpoint_dir + 'args.pkl', 'wb') as fp:
        pickle.dump(args, fp)

    print('Data and model loaded')
    print('Checkpoint dir:', checkpoint_dir)

    for epoch in range(args.num_epochs):
        train(epoch, model, optimizer, checkpoint_dir, loader_train)    #一次训练之后   一次验证
        vald(epoch, model, checkpoint_dir, loader_val)

        writer.add_scalar('trainloss', np.array(metrics['train_loss'])[epoch], epoch)
        writer.add_scalar('valloss', np.array(metrics['val_loss'])[epoch], epoch)

        if args.use_lrschd:     #默认是true
            scheduler.step()

        print('*' * 30)
        print('Epoch:', args.dataset + '/' + args.tag, ":", epoch)
        for k, v in metrics.items():
            if len(v) > 0:
                print(k, v[-1])    #输出 train_loss   val_loss  这个是输出每个epoch之后的最后一个数值

        print(constant_metrics)   #{'min_val_epoch':  'min_val_loss':   'min_train_epoch':  'min_train_loss':  }
        print('*' * 30)

        with open(checkpoint_dir + 'constant_metrics.pkl', 'wb') as fp:
            pickle.dump(constant_metrics, fp)


if __name__ == '__main__':

    # 自定义目录存放日志文件
    log_path = './Logs_train/'
    if not os.path.exists(log_path):
        os.makedirs(log_path)
    # 日志文件名按照程序运行时间设置
    log_file_name = log_path + 'log-' + time.strftime("%Y%m%d-%H%M%S", time.localtime()) + '.log'
    # 记录正常的 print 信息
    sys.stdout = Logger(log_file_name)
    # 记录 traceback 异常信息
    sys.stderr = Logger(log_file_name)

    args = parser.parse_args()
    main(args)
