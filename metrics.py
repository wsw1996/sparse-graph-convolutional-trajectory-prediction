import math
import torch
import numpy as np

def ade(predAll,targetAll,count_):    #[pre_len,N,2]
    All = len(predAll)     #ALL=3
    sum_all = 0 
    for s in range(All):    #0,1,2
        pred = np.swapaxes(predAll[s][:,:count_[s],:],0,1)   #最后这个应该是 （-， 2）----交换维度之后 就是   （2，-）
        target = np.swapaxes(targetAll[s][:,:count_[s],:],0,1)
        
        N = pred.shape[0]
        T = pred.shape[1]     #Tpred
        sum_ = 0 
        for i in range(N):   #0，1
            for t in range(T):
                sum_+=math.sqrt((pred[i,t,0] - target[i,t,0])**2+(pred[i,t,1] - target[i,t,1])**2)    #平方之后再开根号  是二范数
        sum_all += sum_/(N*T)
        
    return sum_all/All


def fde(predAll,targetAll,count_):
    All = len(predAll)
    sum_all = 0 
    for s in range(All):
        pred = np.swapaxes(predAll[s][:,:count_[s],:],0,1)
        target = np.swapaxes(targetAll[s][:,:count_[s],:],0,1)
        N = pred.shape[0]
        T = pred.shape[1]
        sum_ = 0 
        for i in range(N):
            for t in range(T-1,T):
                sum_+=math.sqrt((pred[i,t,0] - target[i,t,0])**2+(pred[i,t,1] - target[i,t,1])**2)
        sum_all += sum_/(N)

    return sum_all/All


def seq_to_nodes(seq_,max_nodes = 88):  #[batch_size(1), 行人数N， x|y ,obs_len]
    seq_ = seq_.squeeze()           #行人数N， x|y ,obs_len  ---[57,2,8]
    seq_ = seq_[:, :2]
    seq_len = seq_.shape[2]         #8
    max_nodes = seq_.shape[0]  # 自己加的

    V = np.zeros((seq_len,max_nodes,2))      #(8,88,2)
    for s in range(seq_len):         #0 1 2 3 4 5 6 7
        step_ = seq_[:,:,s]                    #相当于 8个 【57，2】   #step_是[57,2]八个点，循环一次填每个点的57个行人的 xy
        for h in range(len(step_)):    #这个长度都是2啊,h值为 0 1
            V[s,h,:] = step_[h]        ##v[8,57,2] s对应8，h对应57，     ： 对应xy   ，填进去的x y坐标
            
    return V.squeeze()

def nodes_rel_to_nodes_abs(nodes,init_node):
    nodes = nodes[:, :, :2]
    init_node = init_node[:, :2]
    nodes_ = np.zeros_like(nodes)                   #nodes_: [obs_len N 3]
    for s in range(nodes.shape[0]):          #0-7
        for ped in range(nodes.shape[1]):    #行人数             # np.sum(nodes[:s+1,ped,:],axis=0)是压缩行，多行变成单行
            nodes_[s,ped,:] = np.sum(nodes[:s+1,ped,:],axis=0) + init_node[ped,:]      #obs_len,行人数N，3    0观察长度时 所有行人的 xy坐标
                                                                                       ##init_node【行人数N，2】
    return nodes_.squeeze()
#最后得到的 nodes_[s,ped,:]  就是[obs_len,行人数N，2]

def closer_to_zero(current,new_v):
    dec =  min([(abs(current),current),(abs(new_v),new_v)])[1]
    if dec != current:
        return True
    else: 
        return False
        
def bivariate_loss(V_pred,V_trgt):
    #mux, muy, sx, sy, corr
    #assert V_pred.shape == V_trgt.shape
    # 高斯分布概率密度函数 f(x) =  σ乘以根号下2π分之一  乘以 e 的 负的 x-u 的平方 除以2 σ平方

   # [1 obs_len N 3]
  #  [1 pred_len N 2]    V_trgt:[batch_size,obs_len,num_preds,xy]------[128， 8， 57， 2]
    normx = V_trgt[:,:,0]- V_pred[:,:,0]    #---真实的x值减去预测的x值
    normy = V_trgt[:,:,1]- V_pred[:,:,1]    #---真实的y值减去预测的y值

    # （不是）所以说为什么预测的最后会有五个，因为前两个是x y 坐标  后面三个是参数，用来得出预测值的参数  均值、标准差、相关系数

    sx = torch.exp(V_pred[:,:,2]) #sx
    sy = torch.exp(V_pred[:,:,3]) #sy
    corr = torch.tanh(V_pred[:,:,4]) #corr
    
    sxsy = sx * sy

    z = (normx/sx)**2 + (normy/sy)**2 - 2*((corr*normx*normy)/sxsy)
    negRho = 1 - corr**2

    # 分子 Numerator
    result = torch.exp(-z/(2*negRho))
    # Normalization factor 归一化因子
    denom = 2 * np.pi * (sxsy * torch.sqrt(negRho))

    # Final PDF calculation 最终距离计算
    result = result / denom
    # Numerical stability  数值稳定性
    epsilon = 1e-20   #1*10的负20次方  最小数
    result = -torch.log(torch.clamp(result, min=epsilon))    #设定了最小数，loss最小
    result = torch.mean(result)    #没有dim,就是所值求平均
    
    return result