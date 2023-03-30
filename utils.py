import os
import math
import torch
import numpy as np
import torch.utils.data
from torch.utils.data import Dataset
from tqdm import tqdm

import copy

import pandas as pd
import sys
from termcolor import colored


def anorm(p1, p2):
    NORM = math.sqrt((p1[0] - p2[0]) ** 2 + (p1[1] - p2[1]) ** 2)
    if NORM == 0:
        return 0
    return 1 / (NORM)

def loc_pos(seq_):     #  xy坐标----这个result就是在seq_的基础上   最后一维度上增加了一个而已

    # seq_ [obs_len N 2]  N是行人数

    obs_len = seq_.shape[0]  #序列长度
    num_ped = seq_.shape[1]   #行人数

    pos_seq = np.arange(1, obs_len + 1)  #起点是1 ，终点是obs_len + 1        #一维的  [1，2，3，4，5，6，7，8]  ---shape(8,)
    pos_seq = pos_seq[:, np.newaxis, np.newaxis]   #之前是2维，现在 是4维了        ?     -----shape(8,1,1)
    pos_seq = pos_seq.repeat(num_ped, axis=1)   #复制  行人数  行，复制一列，相当于一个行人一行 ---------shape(8,57,1)

    result = np.concatenate((pos_seq, seq_), axis=-1) #对应行的数组拼接   维度不变  相当于在pos_seq后面添加 seq数据了------shape(8,57,3)---新增的那列的值就是（1，obs_len+1)的随机值--并且是在坐标之前的
                                                     #第三个维度拼接在一起了
    return result     #[obs_len, num_preds,3]

def seq_to_graph(seq_, seq_rel, pos_enc=False):   #  xy坐标   xy坐标差值
    seq_ = seq_.squeeze()    #删掉了一个为1 的维度，例如（2，1，3）变成了（2，3）;若没有1 的，则没有改变.-----[16,2,8]
    seq_rel = seq_rel.squeeze()#   seq_和seq_rel都应该是 4维的啊2785*[57,2,8] 相当于所有帧数中，每个里面的[行人数、坐标、序列长度]，每个序列长度 8  12
    # if len(seq_)!=3:
    #     seq_ = seq_.unsqueeze(0)
    #     seq_len = seq_.shape[2]    #  每个帧中 相当于[1,行人数、坐标、序列长度]，经过删除维度之后，删掉1，剩下的---[2]对应就是序列长度 8  12
    seq_len = seq_.shape[2]
    max_nodes = seq_.shape[0]  # [0]对应于行人数，节点

    V = np.zeros((seq_len, max_nodes, 4))  #8个 行人数 2---[8,N,2]
    for s in range(seq_len):  #0，1，2，3，4，5，6，7                                                          #------seq_[57, 2, 8]
        step_ = seq_[:, :, s]        #seq-是[57,2,8]  step_是[57,2]  八个点，循环一次填 每个点 的57个行人的 x y #------step_[57, 2, 8]
        step_rel = seq_rel[:, :, s]       #----[16,2]                                                         #---- step_rel[57, 2, 8]   xy差值
        for h in range(len(step_)):    #len(step_)是二维的行数=57 len(step_[0])或者len(step_[1])是列数  第s个序列长度的 h个人的x y 坐标都写入了
            V[s, h, :] = step_rel[h]   #v[8,57,2] s对应8，h对应57，     ： 对应x y-------节点

    if pos_enc:
        V = loc_pos(V)           ##[obs_len, num_preds,3]  #3中第0列是 1，2，3，4，5，6，7，8   ---这个result就是在seq_的基础上   最后一维度上增加了一个而已

    return torch.from_numpy(V).type(torch.float)

def poly_fit(traj, traj_len, threshold):
    """
    Input:
    - traj: Numpy array of shape (2, traj_len)  --形状为Numpy的数组（2，traj_len）
    - traj_len: Len of trajectory  -- 轨迹的Len
    - threshold: Minimum error to be considered for non linear traj--非线性轨迹应考虑的最小误差
    Output:
    - int: 1 -> Non Linear 0-> Linear
    """
    t = np.linspace(0, traj_len - 1, traj_len)
    res_x = np.polyfit(t, traj[0, -traj_len:], 2, full=True)[1]  #t和traj[0, -traj_len:]长度必须是一样的，但是我在窗口测试的时候不行，因为没有循环，
    res_y = np.polyfit(t, traj[1, -traj_len:], 2, full=True)[1]  #导致curr_ped_seq[0, -12:]为 array([1.28])，只有一个值，正常的应该是都有12个点（预测的长度）
    if res_x + res_y >= threshold:             #这里的x y  就是对应位置坐标x y
        return 1.0     #x y 坐标大于阈值  就是线性的呗，列表中添加1
    else:
        return 0.0      #否则就是非线性的了，  阈值是最小误差


# def read_file(_path, delim='\t'):
#     data = []
#     if delim == 'tab':
#         delim = '\t'
#     elif delim == 'space':
#         delim = ' '
#     with open(_path, 'r') as f:
#         for line in f:
#             line = line.strip().split(delim)
#             line = [float(i) for i in line]
#             data.append(line)
#     return np.asarray(data)  #将data转换为array

# def read_file(_path):
#     data = []
#
#     with open(_path, 'r') as f:
#         for line in f:
#             line = line.strip().split()
#             line = [float(i) for i in line]
#             data.append(line)
#     return np.asarray(data)  #将data转换为array

class TrajectoryDataset(Dataset):
    """Dataloder for the Trajectory datasets"""

    def __init__(
            self, data_dir, obs_len=10, pred_len=30, skip=1, threshold=0.002,
            min_ped=1, delim='\t'):
        """
        -data_dir：目录包含格式为的数据集文件，
        <frame_id> <ped_id> <x> <y>
        - obs_len: 输入轨迹中的时间步数，
        - pred_len: 输出轨迹中的时间步数，
        - skip: 制作数据集时要跳过的帧数，
        - threshold: 非线性轨迹应考虑的最小误差，
        使用线性预测器时：
        - min_ped: 应连续行驶的最小行人数量
        - delim: 数据集文件中的定界符，

        Args:
        - data_dir: Directory containing dataset files in the format
        <frame_id> <ped_id> <x> <y>
        - obs_len: Number of time-steps in input trajectories
        - pred_len: Number of time-steps in output trajectories
        - skip: Number of frames to skip while making the dataset
        - threshold: Minimum error to be considered for non linear traj
        when using a linear predictor
        - min_ped: Minimum number of pedestrians that should be in a seqeunce
        - delim: Delimiter in the dataset files
        """
        super(TrajectoryDataset, self).__init__()

        self.max_peds_in_frame = 0
        self.data_dir = data_dir  #dataset\args.dataset\train
        self.obs_len = obs_len
        self.pred_len = pred_len
        self.skip = skip
        self.seq_len = self.obs_len + self.pred_len
        self.delim = delim

        all_files = os.listdir(self.data_dir)#显示的就是 train\test\val下面的七个txt文件，索引为0-6
        all_files = [os.path.join(self.data_dir, _path) for _path in all_files]  #_选定7个文件中的一个了

        # print(all_files)

        num_peds_in_seq = []
        seq_list = []         #seq_list 四维？
        seq_list_rel = []    #
        loss_mask_list = []
        non_linear_ped = []

        for path in all_files:
            # data = read_file(path, delim)  #读七个txt文件中的一个文件,按行读取数据，然后转换为数组
            #print("data",data)读出来  train时，train文件下的7个txt文件
            #test 时，test文件文件下的7个txt文件
            # print("dddddd:",data.shape)  #对于path="dataset/eth/train/biwi_hotel_train.txt"来说的（4946，4）

            data = pd.read_csv(path)
            data = np.asarray(data)
            data = data[:, :6]
            print("data_vessel:",data.shape)


            frames = np.unique(data[:, 0]).tolist()#去重后 一共有多少帧---只看第一列，是帧数，去掉重复的帧数，剩下934长度,[0.0, 10.0, 20.0, 30.0, 40.0, 50....14380.0, 14390.0]

            frame_data = []
            for frame in frames:   #帧数长度是934
                frame_data.append(data[frame == data[:, 0], :])   #frame_data是得到了去掉重帧数之后的数据，按帧取出所有数据-生成列表
            # print("framme shape:",frame_data)# 我感觉到这里应该是一样的啊
            num_sequences = int(
                math.ceil((len(frames) - self.seq_len + 1) / skip))  #啥意思? #向上取整， 减去序列长度再加1   915

            for idx in range(0, num_sequences * self.skip + 1, skip):#所有帧数？  0- 915*1+1   idx为0，1，，，，915-----下面控制台打印出来的结果都是idx=915时候的结果哈
                curr_seq_data = np.concatenate(frame_data[idx:idx + self.seq_len], axis=0)#取8+12共20作为一个序列，915个20的序列eg:0-20帧所有行人的所有坐标，把这个帧数范围内的所有的四列数据都进行拼接起来-（61,4）
                # print("curr_seq_data:",curr_seq_data.shape)  #(211,4)
                #2785个20的序列eg:0-20帧所有行人的所有坐标[20*57*2]   20帧内  行人的xy坐标
                peds_in_curr_seq = np.unique(curr_seq_data[:, 1])#去重取出每个序列里的行人 eg:0-20帧内所有出现的行人---(7,)---由pre_id构成--
                #这步有
                # print("peds_in_curr_seq:", peds_in_curr_seq.shape)
                # print("peds_in_curr_seq_len:", len(peds_in_curr_seq))
#取idx=1, curr_seq_data为（122，4），该段中出现 行人16
                self.max_peds_in_frame = max(self.max_peds_in_frame, len(peds_in_curr_seq))#返回给定参数的最大值，最大行人数自我更新  16   ---7
                curr_seq_rel = np.zeros((len(peds_in_curr_seq), 4, self.seq_len))  #[16,2, 20]   行人、坐标、序列长度   值为0  ----[7,2,20]
                # print("curr_seq_rel:", curr_seq_rel.shape)
                curr_seq = np.zeros((len(peds_in_curr_seq), 4, self.seq_len))     #[16,2, 20]    行人、坐标、序列长度   值为0
                curr_loss_mask = np.zeros((len(peds_in_curr_seq),self.seq_len))   #[16,20]
                num_peds_considered = 0
                _non_linear_ped = []
                for _, ped_id in enumerate(peds_in_curr_seq):#单位帧内所有人16人
                    #这步有
                    curr_ped_seq = curr_seq_data[curr_seq_data[:, 1] ==ped_id, :]#把当前行人的20个序列x y取出[20*2] ? ---在当前帧内的行人 的 坐标取出来---20行xy坐标
                    # print("curr_ped_seq:",curr_ped_seq.shape)
                    curr_ped_seq = np.around(curr_ped_seq, decimals=4) #返回四位小数
                    # print("cucucuccu:",curr_ped_seq.shape)
                    pad_front = frames.index(curr_ped_seq[0, 0]) - idx    #首帧
                    pad_end = frames.index(curr_ped_seq[-1, 0]) - idx + 1   #尾帧
                    # print("ped_end-pad_front",pad_end-pad_front)

                    if pad_end - pad_front != self.seq_len:
                        continue

                    if curr_ped_seq.shape[0] != self.seq_len:
                        continue

                    curr_ped_seq = np.transpose(curr_ped_seq[:, 2:]) #取第二列之后的xy，【ped_id,xy坐标】再转置[2*20] -----这应该是2*20，对应轨迹长度
                    # curr_ped_seq = curr_ped_seq
                    # print("pad_front:", pad_front)
                    # print("pad_end:", pad_end)

                    # Make coordinates relative 建立相对坐标系     这块之前理解错了  已经固定了ped_id  每个行人的 在20帧内的 xy坐标
                    rel_curr_ped_seq = np.zeros(curr_ped_seq.shape)#2*20
                    # ipdb.set_trace()
                    rel_curr_ped_seq[:, 1:] = \
                        curr_ped_seq[:, 1:] - curr_ped_seq[:, :-1]  #下一帧减当前帧坐标——>求出变化量------[2*1]--[2*20]---差值
                    # rel_curr_ped_seq[:, 1:] = \
                    #     curr_ped_seq[:, 1:] - np.reshape(curr_ped_seq[:, 0], (2,1))
                    _idx = num_peds_considered   #0
                    curr_seq[_idx, :, pad_front:pad_end] = curr_ped_seq#把多个行人的序列放在一个列表 ：---[16*2*20]  所有行人id的 20帧内的所有xy坐标
                    # print("curr_ped_seq.shape:",curr_ped_seq.shape)  #(2, 20)
                    # print("curr_seq.shape:",curr_seq.shape)  #(16, 2, 20)
                    curr_seq_rel[_idx, :, pad_front:pad_end] = rel_curr_ped_seq  #把多个行人的序列放在一个列表   ：---[16*2*20]---------差值
                    # Linear vs Non-Linear Trajectory
                    _non_linear_ped.append(
                        poly_fit(curr_ped_seq, pred_len, threshold))#把线性的找出来放在列表里
                    curr_loss_mask[_idx, pad_front:pad_end] = 1  #设置掩码   （16*20）  最小是0，最大是1---[7,20]
                    num_peds_considered += 1

                # print("ped finish select-------")

                if num_peds_considered > min_ped:   #连续行驶的行人数超过1时候,就把他加到线性的行人里面
                    non_linear_ped += _non_linear_ped
                    num_peds_in_seq.append(num_peds_considered) #记录连续行驶的行人  记录有多少人
                    loss_mask_list.append(curr_loss_mask[:num_peds_considered]) #掩码    ---------[16，20]
                    seq_list.append(curr_seq[:num_peds_considered])  #把这个序列所有人的坐标加进列表,得到[2785*57,2,20]-----#[16,2, 20] ---
                    #这个 就是把curr_seq中 原来的0值换成 行人坐标了---  为什么csdn的这个会式4维的，可能因为在循环的帧数内，所以前面加上了帧数
                    seq_list_rel.append(curr_seq_rel[:num_peds_considered])    #这个和上面应该是一样的，只不过这个填的是坐标的差值

        self.num_seq = len(seq_list)
        seq_list = np.concatenate(seq_list, axis=0)  #[2785*57,2,20]---这块才是将所有帧数内的 数据拼接起来吧，是seq_list在concat前就已经是4维了？所以拼接之后第一个维度加起来，得到所有帧数？
        seq_list_rel = np.concatenate(seq_list_rel, axis=0)
        loss_mask_list = np.concatenate(loss_mask_list, axis=0)   #-----[915,16,20]
        non_linear_ped = np.asarray(non_linear_ped)

        # Convert numpy -> Torch Tensor  数组转成张量     seq_list （xy坐标） 和seq_list_rel（差值）维度一样
        self.obs_traj = torch.from_numpy(
            seq_list[:, :, :self.obs_len]).type(torch.float)     #[2785*57,2,8]----[915,16,2,8] ---xy坐标值
        self.pred_traj = torch.from_numpy(
            seq_list[:, :, self.obs_len:]).type(torch.float)     #[2785*57,2,12]  -----[915,16,2,12] ---xy坐标值
        self.obs_traj_rel = torch.from_numpy(
            seq_list_rel[:, :, :self.obs_len]).type(torch.float)     #[2785*57,2,8]--------[915,16,2,8] ---xy坐标差值
        self.pred_traj_rel = torch.from_numpy(
            seq_list_rel[:, :, self.obs_len:]).type(torch.float)      #[2785*57,2,12]   -----[915,16,2,12] ---xy坐标差值
        self.loss_mask = torch.from_numpy(loss_mask_list).type(torch.float)    # #-----[915,16,20]
        self.non_linear_ped = torch.from_numpy(non_linear_ped).type(torch.float)
        cum_start_idx = [0] + np.cumsum(num_peds_in_seq).tolist()  #变成了[0,1,3,4,5.....16]-----行人超过1时，就加进去，所以他这个其实时一帧内行人数大于1时候记录的行人数
        self.seq_start_end = [                                      #累加求和了，因为开始时记录的是一帧内的行人，一个一个添加的，[1,1,1,1...]，然后累加求和得到一个数了。
            (start, end)                                            #一共是915帧，所以这个就有915个 长度。所以第一个取出来的（s,e）就是一帧内行人数
            for start, end in zip(cum_start_idx, cum_start_idx[1:])
        ]   #存储的是所有trajectory集合  #cum_start_idx是当前帧涉及的人数， cum_start_idx[1:]是下一帧的人数
        #cum_start_idx是当前帧内的人数，下一个数[1:]就是下一个帧内的人数
        # Convert to Graphs
        self.v_obs = []
        self.v_pred = []
        print("Processing Data .....")

        pbar = tqdm(total=len(self.seq_start_end))
        for ss in range(len(self.seq_start_end)):   #这个值是915   #len是2785
            pbar.update(1)

            start, end = self.seq_start_end[ss]    ##取出来一个（start,end）  当前帧的人数，下一个帧内的人数  #取出来开始和结束时的人数

           # obs_traj [915, 16, 2, 8]
           #  print("obs----:",self.obs_traj[start:end, :].shape)

            v_= seq_to_graph(self.obs_traj[start:end, :], self.obs_traj_rel[start:end, :], True)  #-----V[obs_len, N, 3]-------输出与obs_traj[start:end, :],一致是xy坐标
            ##self.obs_traj[start:end,:]是2785个seq_list中的1个[57,2,8]，，self.obs_traj是2785*[57,2,8]
            self.v_obs.append(v_.clone())  ##2785*8*57*3  经过2785个循环,v_obs从[8,57,3]变成[2785,8,57,3]
            v_= seq_to_graph(self.pred_traj[start:end, :], self.pred_traj_rel[start:end, :], False)  #所以要注意这里就是 观察的是【obs_len,行人数，3】，
            self.v_pred.append(v_.clone())                                                            #预测的是【obs_len,行人数，2】

        pbar.close()

    def __len__(self):
        return self.num_seq

    def __getitem__(self, index):  #index相当于编号，
        start, end = self.seq_start_end[index]   #得到很多（s,e）对  获得定长的数组

        out = [
            self.obs_traj[start:end, :], self.pred_traj[start:end, :],
            self.obs_traj_rel[start:end, :], self.pred_traj_rel[start:end, :],
            self.non_linear_ped[start:end], self.loss_mask[start:end, :],
            self.v_obs[index], self.v_pred[index]                                          #x y 坐标
        ]    #obs_traj是当前的坐标   obs_traj_rel是差值，下一帧减当前帧坐标——>求出变化量
        return out


# class TrajectoryVesselDataset(Dataset):
#     def __init__(
#             self, data_dir, obs_len, pred_len, skip, feature_size):
#         super(TrajectoryVesselDataset, self).__init__()
#
#         self.data_dir = data_dir  # dataset\args.dataset\train
#         self.obs_len = obs_len
#         self.pred_len = pred_len
#         self.skip = skip
#         self.seq_len = self.obs_len + self.pred_len
#         self.feature_size = feature_size
#
#         self.len = 0
#         self.sequences = {}
#         self.masks = {}
#         self.vesselCount = {}
#         self.shift = self.obs_len
#         filecsv = os.listdir(data_dir)
#
#         for i_dt, dt in enumerate(filecsv):
#             print("%03i / %03i - loading %s"%(i_dt+1,len(filecsv),dt))
#
#         self.filenames = os.path.join(data_dir,dt)
#         print("filenames是什么：",self.filenames)
#         # for f, filename in enumerate(self.filenames):
#         df = self.load_df(self.filenames)
#         print("data size:",df.shape)   #(2371228,4)
#
#         # df = self.normalize(df)
#         if not df.empty:
#             self.get_file_samples(df)
#
#     def load_df(self, filename):
#         # df = pd.read_csv(filename, header=0, usecols=['BaseDateTime', 'MMSI', 'LAT', 'LON', 'SOG', 'Heading'],
#         #                  parse_dates=['BaseDateTime'])
#         df = pd.read_csv(filename, header=0, usecols=['BaseDateTime', 'MMSI', 'LAT', 'LON'],
#                          parse_dates=['BaseDateTime'])
#         df.sort_values(['BaseDateTime'], inplace=True)
#         return df
#
#     def normalize(self, df):
#         df['LAT'] = (math.pi / 180) * df['LAT']
#         df['LON'] = (math.pi / 180) * df['LON']
#         df = df.loc[(df['LAT'] <= max_lat) & (df['LAT'] >= min_lat) & (df['LON'] <= max_lon) & (df['LON'] >= min_lon)]
#         if not df.empty:
#             df['LAT'] = (df['LAT'] - min_lat) / (max_lat - min_lat)
#             df['LON'] = (df['LON'] - min_lon) / (max_lon - min_lon)
#             df['SOG'] = df['SOG'] / 22
#             df['Heading'] = df['Heading'] / 360
#             print(df['SOG'].min(), df['SOG'].max())
#         return df
#
#     def get_file_samples(self, df):
#         j = 0
#         timestamps = df['BaseDateTime'].unique()
#         while not (j + self.obs_len + self.pred_len) > len(timestamps):
#             frame_timestamps = timestamps[j:j + self.obs_len + self.pred_len]  #siz是20，20个不重复的帧数
#             frame = df.loc[df['BaseDateTime'].isin(frame_timestamps)]   #把这些帧数对应的所有datatime数据都取进来，所以这个frame一定超过20了
#             if self._condition_time(frame_timestamps):
#                 cond_val, vessels = self._condition_vessels(frame)   #condition_satisfied, total_vessels  得到了20帧内的船数量
#                 if cond_val:
#                     sys.stdout.write(colored("\rfile: {}/{} Sample: {} Num Vessels: {}".format(1, len(self.filenames), self.len, vessels), "blue"))
#                     self.sequences[self.len], self.masks[self.len], self.vesselCount[self.len] = self.get_sequence(frame)
#                     self.len += 1
#             j += self.shift
#
#     def _condition_time(self, timestamps):
#         condition_satisfied = True
#         diff_timestamps = np.amax(np.diff(timestamps).astype('float'))  # 时间的差值
#         if diff_timestamps / (6e+10) > 1 or diff_timestamps / (8.64e+13) >= 1:
#             condition_satisfied = False
#         return condition_satisfied
#
#     def _condition_vessels(self, frame):
#         condition_satisfied = True
#         frame_timestamps = frame['BaseDateTime'].unique()[:self.obs_len]
#         frame = frame.loc[frame['BaseDateTime'].isin(frame_timestamps)]
#         total_vessels = len(frame['MMSI'].unique())
#         valid_vessels = [v for v in frame['MMSI'].unique() if not \
#         abs(frame.loc[frame['MMSI'] == v]['LAT'].diff()).max() < (1e-04) \
#         and not abs(frame.loc[frame['MMSI'] == v]['LON'].diff()).max() < (1e-04) and len(frame.loc[frame['MMSI'] == v]) == self.obs_len]
#
#         return condition_satisfied, total_vessels
#
#     def get_sequence(self, frame):  #frame就是0-20帧内的
#         frame = frame.values
#         frameIDs = np.unique(frame[:, 0]).tolist()    #20个不同的帧数
#         input_frame = frame[np.isin(frame[:, 0], frameIDs[:self.obs_len])]  #返回true,false  为了mask?   （4883，6） [8]  ===（3876，6）
#         vessels = np.unique(input_frame[:, 1]).tolist()    #8帧内的 船数
#         sequence = torch.FloatTensor(len(vessels), len(frameIDs), frame.shape[-1] - 2)   # 930(船数), 20（帧数）,   6-2=4  四维的
#         mask = torch.BoolTensor(len(vessels), len(frameIDs))   # 930 * 20
#         for v, vessel in enumerate(vessels):
#             vesselTraj = frame[frame[:, 1] == vessel]   #frame是帧数、MMSI，特征（4个），所以这个是取得8帧内的船相对应的 数据-------（3，6）
#             vesselTrajLen = np.shape(vesselTraj)[0]  #   帧数   3
#             vesselIDs = np.unique(vesselTraj[:, 0])   #这是让一个船只有一个帧？  3
#             maskVessel = np.ones(len(frameIDs))       #（20，）
#             if vesselTrajLen < (self.obs_len + self.pred_len):
#                 missingIDs = [f for f in frameIDs if not f in vesselIDs]   #对于在20里面 但是不在3里面的   9个
#                 maskVessel[vesselTrajLen:].fill(0.0)    #将填充的船只用0填充   （20，）
#                 paddedTraj = np.zeros((len(missingIDs), np.shape(vesselTraj)[1]))   #----9， （3，6）---6 --所以填充的轨迹是（9*6）
#                 vesselTraj = np.concatenate((vesselTraj, paddedTraj), axis=0)       #  -----拼接之后变成（20，6）
#                 vesselTraj[vesselTrajLen:, 0] = missingIDs
#                 vesselTraj[vesselTrajLen:, 1] = vessel * np.ones((len(missingIDs)))     #17，930*（17，）
#                 sorted_idx = vesselTraj[:, 0].argsort()           #按照帧数排一下  (20,)   0-19
#                 vesselTraj = vesselTraj[sorted_idx, :]           #(20,6)
#                 maskVessel = maskVessel[sorted_idx]              #(20,)
#                 vesselTraj[:, 2:] = fillarr(vesselTraj[:, 2:])    #20帧内 后面的四个特征
#             vesselTraj = vesselTraj[:, 2:]                       #(20,4)
#             sequence[v, :] = torch.from_numpy(vesselTraj.astype('float32'))        #930, (20,4)
#             mask[v, :] = torch.from_numpy(maskVessel.astype('float32')).bool()      #930, (20,4)
#         vessel_count = torch.tensor(len(vessels))                                 #930   是 20帧内船的数量
#         return sequence, mask, vessel_count
#
#     def __getitem__(self, idx):
#         idx = int(idx.numpy()) if not isinstance(idx, int) else idx
#         sequence, mask, vessel_count = self.sequences[idx], self.masks[idx], self.vesselCount[idx]
#         ip = sequence[:, :self.obs_len, ...]   #输入  (930,0-8,...)---（930，8，4）
#         op = sequence[:, self.obs_len:, ...]   #输出   (930,12-20,...)
#         distance_matrix, bearing_matrix, heading_matrix = get_features(ip, 0)  #(930,8,930)
#         ip_mask = mask[:, :self.obs_len]      #(930,8)-----（930，8，4）
#         op_mask = mask[:, self.obs_len:]      #(930,12)-----（930，12，4）
#         ip = ip[..., :self.feature_size]        #feature_size=2  这块是只输入了经纬度呗 相当于 （930，8，2）
#         op = op[..., :self.feature_size]        #（930，12，2）
#
#         print('input:', ip)
#         print('output:',op)
#         print('distance_matrix', distance_matrix)
#         print('bearing_matrix',bearing_matrix)
#         print( 'heading_matrix',heading_matrix)
#         print('input_mask',ip_mask)
#         print('output_mask',op_mask)
#         print( 'vessels',vessel_count)
#         return {'input': ip, \
#                 'output': op, \
#                 'distance_matrix': distance_matrix, \
#                 'bearing_matrix': bearing_matrix, \
#                 'heading_matrix': heading_matrix, \
#                 'input_mask': ip_mask, \
#                 'output_mask': op_mask, \
#                 'vessels': vessel_count}
#
#
#
#
#     def __len__(self):
#         return self.len
#
#
# def get_features(sample, neighbors_dim, previous_sample=None):    #ip 0
# 	if not (neighbors_dim==1):
# 		sample = sample.transpose(0,1)     #  (8,930,4)
# 	theta_1 = get_heading(sample,previous_sample)    #  (8,930)?
# 	theta_1 = theta_1.unsqueeze(-1)                  #  (8,930,1)
# 	lat_1, lon_1 = sample[...,0], sample[...,1]      #(8,930,)  (8,930,)就是（8，930）
# 	lat_1, lon_1 = lat_1.unsqueeze(-1).expand(sample.size(0),sample.size(1),sample.size(1)), lon_1.unsqueeze(-1).expand(sample.size(0),sample.size(1),sample.size(1))  #最后是expand为（8，930，930）
# 	lat_2, lon_2 = lat_1.transpose(1,2), lon_1.transpose(1,2)    #（8，930，930）
# 	distance = equirectangular_distance(lat_1, lon_1, lat_2, lon_2)    #（8，930，930）？
# 	bearing = absolute_bearing(lat_1, lon_1, lat_2, lon_2)                  #（8，930，930）？
# 	theta_1 = theta_1.expand(sample.size(0),sample.size(1),sample.size(1))   ##（8，930，930）
# 	theta_2 = theta_1.transpose(1,2)                                         ##（8，930，930）
# 	bearing = bearing-theta_1                    #（8，930，930）
# 	bearing[distance==distance.min()]=0
# 	bearing[bearing<0]+=360              #compass_bearing = (initial_bearing + 360) % 360   为了归一化
# 	heading = theta_2-theta_1
# 	heading[heading<0]+=360
# 	if not (neighbors_dim==1):
# 		distance, bearing, heading = distance.transpose(0,1), bearing.transpose(0,1), heading.transpose(0,1)  #(930,8,930)
# 	return distance, bearing, heading
#
#
# radius_earth = 3440.1
# def equirectangular_distance(lat1, lon1, lat2, lon2):
# 	lat1, lon1 = scale_values(lat1, lon1)
# 	lat2, lon2 = scale_values(lat2, lon2)
# 	dlon = lon2-lon1
# 	dlat = lat2-lat1
# 	dist = (dlat)**2 + (dlon*torch.cos((lat1+lat2)/2))**2
# 	dist = radius_earth*torch.sqrt(dist+(1e-24))
# 	return dist
#
# def absolute_bearing(lat1, lon1, lat2, lon2):   #绝对方位，利用经纬度求  得到两个点之间的方位
# 	lat1, lon1 = scale_values(lat1, lon1)
# 	lat2, lon2 = scale_values(lat2, lon2)
# 	dlon = lon2-lon1
# 	y = torch.sin(dlon)*torch.cos(lat2)
# 	x = torch.cos(lat1)*torch.sin(lat2) - torch.sin(lat1)*torch.cos(lat2)*torch.cos(dlon)
# 	x = x + (1e-15)
# 	y = y + (1e-15)
# 	bearing = torch.atan2(y,x)
# 	bearing = rad2deg(bearing)  #弧度转换为角度 math.degrees(initial_bearing)
# 	bearing[bearing<0]+=360        #compass_bearing = (initial_bearing + 360) % 360   为了归一化
# 	return bearing
#
# def rad2deg(point):
# 	point = (180/math.pi)*point
# 	return point
#
# # min_lat, max_lat, min_lon, max_lon = 32, 35, -120, -117
# min_lat, max_lat, min_lon, max_lon = 25, 35, -122, -112
#
# min_lat , max_lat, min_lon, max_lon  = (math.pi/180)*min_lat, (math.pi/180)*max_lat, (math.pi/180)*min_lon, (math.pi/180)*max_lon
# def scale_values(lat, lon):
# 	lat = (max_lat-min_lat)*lat + min_lat
# 	lon = (max_lon-min_lon)*lat + min_lon
# 	return lat, lon
#
# def get_heading(sample, prev_sample=None):   #sample:(8,930,4)
# 	n = sample.size(1)
# 	if (prev_sample is None):
# 		if len(sample.size())==3:
# 			prev_sample = sample[:-1,...]    # 不要最后一个
# 			sample = sample[1:,...]           #不要 第一个
# 			heading = absolute_bearing(prev_sample[...,0],prev_sample[...,1],sample[...,0],sample[...,1])  #得到两个点之间的方位
# 			heading = torch.cat((heading[0,...].clone().unsqueeze(0),heading),dim=0)  #heading(8,930)?--
# 		else:
# 			prev_sample = sample[:,:,:-1,...]
# 			sample = sample[:,:,1:,...]
# 			heading = absolute_bearing(prev_sample[...,0],prev_sample[...,1],sample[...,0],sample[...,1])
# 			heading = torch.cat((heading[:,:,0,...].clone().unsqueeze(2),heading),dim=2)
# 	else:
# 		heading = absolute_bearing(prev_sample[...,0],prev_sample[...,1],sample[...,0],sample[...,1])
# 	return heading
#
# def fillarr(arr):    #（20，4）
# 	for i in range(arr.shape[1]):   #0 1 2 3
# 		idx = np.arange(arr.shape[0])    #(20,)
# 		idx[arr[:,i]==0] = 0
# 		np.maximum.accumulate(idx, axis = 0, out=idx)
# 		arr[:,i] = arr[idx,i]
# 		if (arr[:,i]==0).any():    #元素是0，any之后就是False
# 			idx[arr[:,i]==0] = 0
# 			np.minimum.accumulate(idx[::-1], axis=0)[::-1]
# 			arr[:,i] = arr[idx,i]
# 	return arr
#
#
# class TrajVesselDataset(Dataset):
#     """Dataloder for the Trajectory datasets"""
#
#     def __init__(
#             self, data_dir, obs_len=8, pred_len=8, skip=1, threshold=0.002,
#             min_ped=1, delim='\t'):
#         """
#         -data_dir：目录包含格式为的数据集文件，
#         <frame_id> <ped_id> <x> <y>
#         - obs_len: 输入轨迹中的时间步数，
#         - pred_len: 输出轨迹中的时间步数，
#         - skip: 制作数据集时要跳过的帧数，
#         - threshold: 非线性轨迹应考虑的最小误差，
#         使用线性预测器时：
#         - min_ped: 应连续行驶的最小行人数量
#         - delim: 数据集文件中的定界符，
#
#         Args:
#         - data_dir: Directory containing dataset files in the format
#         <frame_id> <ped_id> <x> <y>
#         - obs_len: Number of time-steps in input trajectories
#         - pred_len: Number of time-steps in output trajectories
#         - skip: Number of frames to skip while making the dataset
#         - threshold: Minimum error to be considered for non linear traj
#         when using a linear predictor
#         - min_ped: Minimum number of pedestrians that should be in a seqeunce
#         - delim: Delimiter in the dataset files
#         """
#         super(TrajVesselDataset, self).__init__()
#
#
#
#         # data_set = './vessels_dataset/' + args.dataset + '/'
#
#         # dset_train = TrajectoryVesselDataset(
#         #     data_set + 'train/',
#         #     obs_len=obs_seq_len,
#         #     pred_len=pred_seq_len,
#         #     feature_size=feature_size,
#         #     skip=1)
#         # self.data_dir = data_set + 'train/'
#         self.data_dir = data_dir
#
#         self.obs_len = obs_len
#         self.pred_len = pred_len
#         self.seq_len = self.obs_len + self.pred_len
#         self.skip = skip
#         self.max_peds_in_frame = 0
#
#         num_peds_in_seq = []
#         seq_list = []  # seq_list 四维？
#         seq_list_rel = []  #
#         loss_mask_list = []
#         non_linear_ped = []
#
#         # filecsv = os.listdir(self.data_dir)
#         all_files = os.listdir(self.data_dir)  # 显示的就是 train\test\val下面的七个txt文件，索引为0-6
#         all_files = [os.path.join(self.data_dir, _path) for _path in all_files]  # _选定7个文件中的一个了
#
#         print(all_files)
#
#         # for i_dt, dt in enumerate(filecsv):
#         #     print("%03i / %03i - loading %s" % (i_dt + 1, len(filecsv), dt))
#         #
#         # filename = os.path.join(data_dir, dt)
#         # print("filenames是什么：", filename)
#
#         for path in all_files:
#             # data = read_file(path, delim)  #读七个txt文件中的一个文件,按行读取数据，然后转换为数组
#             # print("data",data)读出来  train时，train文件下的7个txt文件
#             # test 时，test文件文件下的7个txt文件
#             # print("dddddd:",data.shape)  #对于path="dataset/eth/train/biwi_hotel_train.txt"来说的（4946，4）
#
#             data = pd.read_csv(path, header=1)
#             data = np.asarray(data)
#             data = data[:, :4]
#             print("DDDDDDDDD:",data.shape)
#
#             frames = np.unique(data[:, 0]).tolist()  # 去重后 一共有多少帧---只看第一列，是帧数，去掉重复的帧数，剩下934长度,[0.0, 10.0, 20.0, 30.0, 40.0, 50....14380.0, 14390.0]
#
#             frame_data = []
#
#             for frame in frames:  # 帧数长度是934
#                 frame_data.append(data[frame == data[:, 0], :])  # frame_data是得到了去掉重帧数之后的数据，按帧取出所有数据-生成列表
#             # print("framme shape:",frame_data)# 我感觉到这里应该是一样的啊
#             num_sequences = int(
#                 math.ceil((len(frames) - self.seq_len + 1) / skip))  # 啥意思? #向上取整， 减去序列长度再加1   915
#
#             for idx in range(0, num_sequences * self.skip + 1, skip):  # 所有帧数？  0- 915*1+1   idx为0，1，，，，915-----下面控制台打印出来的结果都是idx=915时候的结果哈
#                 curr_seq_data = np.concatenate(frame_data[idx:idx + self.seq_len],
#                                                axis=0)  # 取8+12共20作为一个序列，915个20的序列eg:0-20帧所有行人的所有坐标，把这个帧数范围内的所有的四列数据都进行拼接起来-（61,4）
#                 print("number %s, curr_seq_data: %s" %(idx, curr_seq_data.shape))  # (211,4)
#                 # 2785个20的序列eg:0-20帧所有行人的所有坐标[20*57*2]   20帧内  行人的xy坐标
#                 peds_in_curr_seq = np.unique(curr_seq_data[:, 1])  # 去重取出每个序列里的行人 eg:0-20帧内所有出现的行人---(7,)---由pre_id构成--
#                 # 这步有
#                 # print("peds_in_curr_seq:", peds_in_curr_seq.shape)
#                 # print("peds_in_curr_seq_len:", len(peds_in_curr_seq))
#                 # 取idx=1, curr_seq_data为（122，4），该段中出现 行人16
#                 self.max_peds_in_frame = max(self.max_peds_in_frame,
#                                              len(peds_in_curr_seq))  # 返回给定参数的最大值，最大行人数自我更新  16   ---7
#                 curr_seq_rel = np.zeros(
#                     (len(peds_in_curr_seq), 2, self.seq_len))  # [16,2, 20]   行人、坐标、序列长度   值为0  ----[7,2,20]
#                 # print("curr_seq_rel:", curr_seq_rel.shape)
#                 curr_seq = np.zeros((len(peds_in_curr_seq), 2, self.seq_len))  # [16,2, 20]    行人、坐标、序列长度   值为0
#                 curr_loss_mask = np.zeros((len(peds_in_curr_seq), self.seq_len))  # [16,20]
#                 num_peds_considered = 0
#                 _non_linear_ped = []
#                 for _, ped_id in enumerate(peds_in_curr_seq):  # 单位帧内所有人16人
#                     # 这步有
#                     curr_ped_seq = curr_seq_data[curr_seq_data[:, 1] == ped_id,
#                                    :]  # 把当前行人的20个序列x y取出[20*2] ? ---在当前帧内的行人 的 坐标取出来---20行xy坐标
#                     print("ped_num: %s, curr_ped_seq: %s" %(ped_id, curr_ped_seq.shape))
#                     curr_ped_seq = np.around(curr_ped_seq, decimals=4)  # 返回四位小数
#                     # print("cucucuccu:",curr_ped_seq.shape)
#                     pad_front = frames.index(curr_ped_seq[0, 0]) - idx  # 首帧
#                     pad_end = frames.index(curr_ped_seq[-1, 0]) - idx + 1  # 尾帧
#                     print("ppp_num: %s, ped_end-pad_front:%s" %(ped_id,pad_end - pad_front))
#                     # if pad_end - pad_front != self.seq_len:
#                     #     continue
#                     #
#                     if curr_ped_seq.shape[0] !=self.seq_len:
#                         continue
#
#                     # print("pad_front:", pad_front)  # 0
#                     # print("pad_end:", pad_end)  # 20
#                     # print("curr_pad_seq:", curr_ped_seq.shape)  # (2,11)  (2,17)
#                     curr_ped_seq = np.transpose(curr_ped_seq[:, 2:])  # (11,2)ttrans成（2，11）
#                     print("ped_num: %s,curr_ped_seq.shape:%s" %(ped_id,curr_ped_seq.shape))  # (2, 20)
#                     print("ped_num: %s,curr_seq.shape:%s" % (ped_id,curr_seq.shape))  # (16, 2, 20)
#                     # print("curr_ped_seq start:",curr_ped_seq)  #(2,11)  (2,17)
#                     # curr_ped_seq = curr_ped_seq
#                     rel_curr_ped_seq = np.zeros(curr_ped_seq.shape)  # （2，11）
#                     rel_curr_ped_seq[:, 1:] = curr_ped_seq[:, 1:] - curr_ped_seq[:, :-1]
#                     _idx = num_peds_considered  # 0
#
#                     # 改动
#                     # mask1 = np.zeros((2, self.seq_len))
#                     # curr_ped_seq_mask = copy.deepcopy(mask1)
#                     # curr_ped_seq_mask[:curr_ped_seq.shape[0], :curr_ped_seq.shape[1]] = curr_ped_seq
#                     #
#                     # mask2 = np.zeros((2, self.seq_len))
#                     # rel_curr_ped_seq_mask = copy.deepcopy(mask2)
#                     # rel_curr_ped_seq_mask[:rel_curr_ped_seq.shape[0], :rel_curr_ped_seq.shape[1]] = rel_curr_ped_seq
#
#                     curr_seq[_idx, :, pad_front:pad_end] = curr_ped_seq ##curr_seq(930, 2, 20)
#                     # print("cuee_seq:",curr_seq.shape)
#                     curr_seq_rel[_idx, :, pad_front:pad_end] = rel_curr_ped_seq
#                     # _non_linear_ped.append(
#                     #     poly_fit(curr_ped_seq_mask, self.pred_len, threshold))
#                     curr_loss_mask[_idx, pad_front:pad_end] = 1
#                     # num_peds_considered += 1
#                     # if num_peds_considered > min_ped:
#                     # non_linear_ped += _non_linear_ped
#                 # num_peds_in_seq.append(num_peds_considered)
#                 # loss_mask_list.append(curr_loss_mask[:num_peds_considered])
#                 # seq_list.append(curr_seq[:num_peds_considered])
#                 # seq_list_rel.append(curr_seq_rel[:num_peds_considered])
#             print("data pre fin")
#
#             self.num_seq = len(seq_list)
#             print("num_pred:", self.num_seq)
#
#             seq_list = np.concatenate(curr_seq, axis=0)
#             seq_list_rel = np.concatenate(curr_seq_rel, axis=0)
#             loss_mask_list = np.concatenate(curr_loss_mask, axis=0)
#
#             # seq_list = np.concatenate(seq_list, axis=0)
#             # seq_list_rel = np.concatenate(seq_list_rel, axis=0)
#             # loss_mask_list = np.concatenate(loss_mask_list, axis=0)
#             # non_linear_ped = np.asarray(non_linear_ped)
#
#             # Convert numpy -> Torch Tensor  数组转成张量     seq_list （xy坐标） 和seq_list_rel（差值）维度一样
#             self.obs_traj = torch.from_numpy(
#                 seq_list[:, :, :self.obs_len]).type(torch.float)  # [2785*57,2,8]----[915,16,2,8] ---xy坐标值
#             self.pred_traj = torch.from_numpy(
#                 seq_list[:, :, self.obs_len:]).type(torch.float)  # [2785*57,2,12]  -----[915,16,2,12] ---xy坐标值
#             self.obs_traj_rel = torch.from_numpy(
#                 seq_list_rel[:, :, :self.obs_len]).type(torch.float)  # [2785*57,2,8]--------[915,16,2,8] ---xy坐标差值
#             self.pred_traj_rel = torch.from_numpy(
#                 seq_list_rel[:, :, self.obs_len:]).type(torch.float)  # [2785*57,2,12]   -----[915,16,2,12] ---xy坐标差值
#             self.loss_mask = torch.from_numpy(loss_mask_list).type(torch.float)  # #-----[915,16,20]
#             self.non_linear_ped = torch.from_numpy(non_linear_ped).type(torch.float)
#             cum_start_idx = [0] + np.cumsum(
#                 num_peds_in_seq).tolist()  # 变成了[0,1,3,4,5.....16]-----行人超过1时，就加进去，所以他这个其实时一帧内行人数大于1时候记录的行人数
#             self.seq_start_end = [  # 累加求和了，因为开始时记录的是一帧内的行人，一个一个添加的，[1,1,1,1...]，然后累加求和得到一个数了。
#                 (start, end)  # 一共是915帧，所以这个就有915个 长度。所以第一个取出来的（s,e）就是一帧内行人数
#                 for start, end in zip(cum_start_idx, cum_start_idx[1:])
#             ]  # 存储的是所有trajectory集合  #cum_start_idx是当前帧涉及的人数， cum_start_idx[1:]是下一帧的人数
#             # cum_start_idx是当前帧内的人数，下一个数[1:]就是下一个帧内的人数
#             # Convert to Graphs
#             self.v_obs = []
#             self.v_pred = []
#             print("Processing Data ...PPPPPPP--------..")
#             pbar = tqdm(total=len(self.seq_start_end))
#             for ss in range(len(self.seq_start_end)):  # 这个值是915   #len是2785
#                 pbar.update(1)
#
#                 start, end = self.seq_start_end[ss]  ##取出来一个（start,end）  当前帧的人数，下一个帧内的人数  #取出来开始和结束时的人数
#
#                 # obs_traj [915, 16, 2, 8]
#
#                 v_ = seq_to_graph(self.obs_traj[start:end, :], self.obs_traj_rel[start:end, :],
#                                   True)  # -----V[obs_len, N, 3]-------输出与obs_traj[start:end, :],一致是xy坐标
#                 ##self.obs_traj[start:end,:]是2785个seq_list中的1个[57,2,8]，，self.obs_traj是2785*[57,2,8]
#                 self.v_obs.append(v_.clone())  ##2785*8*57*3  经过2785个循环,v_obs从[8,57,3]变成[2785,8,57,3]
#                 v_ = seq_to_graph(self.pred_traj[start:end, :], self.pred_traj_rel[start:end, :],
#                                   False)  # 所以要注意这里就是 观察的是【obs_len,行人数，3】，
#                 self.v_pred.append(v_.clone())  # 预测的是【obs_len,行人数，2】
#
#             pbar.close()
#
#
#
#     def __len__(self):
#         return self.num_seq
#
#     def __getitem__(self, index):  #index相当于编号，
#         start, end = self.seq_start_end[index]   #得到很多（s,e）对  获得定长的数组
#
#         out = [
#             self.obs_traj[start:end, :], self.pred_traj[start:end, :],
#             self.obs_traj_rel[start:end, :], self.pred_traj_rel[start:end, :],
#             self.non_linear_ped[start:end], self.loss_mask[start:end, :],
#             self.v_obs[index], self.v_pred[index]                                          #x y 坐标
#         ]    #obs_traj是当前的坐标   obs_traj_rel是差值，下一帧减当前帧坐标——>求出变化量
#         return out
#
#
