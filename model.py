import torch
import torch.nn as nn
from torch.nn import functional as F
import numpy as np

class AsymmetricConvolution(nn.Module):

    def __init__(self, in_cha, out_cha):
        super(AsymmetricConvolution, self).__init__()

        self.conv1 = nn.Conv2d(in_cha, out_cha, kernel_size=(3, 1), padding=(1, 0), bias=False)#如果padding=(1,1)中第一个参数表示在高度上面的padding,第二个参数表示在宽度上面的padding
        self.conv2 = nn.Conv2d(in_cha, out_cha, kernel_size=(1, 3), padding=(0, 1))

        self.shortcut = lambda x: x

        if in_cha != out_cha:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_cha, out_cha, 1, bias=False)
            )

        self.activation = nn.PReLU()

    def forward(self, x):

        shortcut = self.shortcut(x)

        x1 = self.conv1(x)
        x2 = self.conv2(x)
        x2 = self.activation(x2 + x1)

        return x2 + shortcut


class InteractionMask(nn.Module):

    def __init__(self, number_asymmetric_conv_layer=2, spatial_channels=4, temporal_channels=4):
        super(InteractionMask, self).__init__()

        self.number_asymmetric_conv_layer = number_asymmetric_conv_layer

        self.spatial_asymmetric_convolutions = nn.ModuleList()
        self.temporal_asymmetric_convolutions = nn.ModuleList()

        for i in range(self.number_asymmetric_conv_layer):
            self.spatial_asymmetric_convolutions.append(
                AsymmetricConvolution(spatial_channels, spatial_channels)
            )
            self.temporal_asymmetric_convolutions.append(
                AsymmetricConvolution(temporal_channels, temporal_channels)
            )

        self.spatial_output = nn.Sigmoid()
        self.temporal_output = nn.Sigmoid()

    def forward(self, dense_spatial_interaction, dense_temporal_interaction, threshold=0.5):

        assert len(dense_temporal_interaction.shape) == 4       #【T，num_heads,N,N】
        assert len(dense_spatial_interaction.shape) == 4       #   (N num_heads T T)

        for j in range(self.number_asymmetric_conv_layer):
            dense_spatial_interaction = self.spatial_asymmetric_convolutions[j](dense_spatial_interaction)
            dense_temporal_interaction = self.temporal_asymmetric_convolutions[j](dense_temporal_interaction)

        spatial_interaction_mask = self.spatial_output(dense_spatial_interaction)
        temporal_interaction_mask = self.temporal_output(dense_temporal_interaction)

        spatial_zero = torch.zeros_like(spatial_interaction_mask, device='cuda')
        temporal_zero = torch.zeros_like(temporal_interaction_mask, device='cuda')   #0

        spatial_interaction_mask = torch.where(spatial_interaction_mask > threshold, spatial_interaction_mask,spatial_zero)
#大于阈值就是激活之后的interaction_mask本身，小于阈值是0
        temporal_interaction_mask = torch.where(temporal_interaction_mask > threshold, temporal_interaction_mask,temporal_zero)

        return spatial_interaction_mask, temporal_interaction_mask


class ZeroSoftmax(nn.Module):

    def __init__(self):
        super(ZeroSoftmax, self).__init__()

    def forward(self, x, dim=0, eps=1e-5):
        x_exp = torch.pow(torch.exp(x) - 1, exponent=2)
        x_exp_sum = torch.sum(x_exp, dim=dim, keepdim=True)
        x = x_exp / (x_exp_sum + eps)
        return x


class SelfAttention(nn.Module):

    def __init__(self, in_dims=4, d_model=64, num_heads=4):     #d_model表示嵌入向量维度（数据嵌入的维度）
        super(SelfAttention, self).__init__()

        self.embedding = nn.Linear(in_dims, d_model)
        self.query = nn.Linear(d_model, d_model)
        self.key = nn.Linear(d_model, d_model)

        self.scaled_factor = torch.sqrt(torch.Tensor([d_model])).cuda()
        self.softmax = nn.Softmax(dim=-1)

        self.num_heads = num_heads

    def split_heads(self, x):

        # x [batch_size seq_len d_model]    #query  和 key的维度都是  [batch_size, seq_len, d_model]  [batch_size , 序列长度， 嵌入维度]，多头是将embedding 分成h份
         #多头是将embedding 分成h份，此时size变成[batch_size, seq_len, h, embeeding/h]
        x = x.reshape(x.shape[0], -1, self.num_heads, x.shape[-1] // self.num_heads).contiguous()   #使用contiguous定义这个的x改变了，但是不影响参数里面的x值
        #所以里面的意思就是：batch_size, seq_len, h, embeeding/h----------将seq_len与h进行转置，为了方便后面的计算
        return x.permute(0, 2, 1, 3)  # [batch_size nun_heads seq_len depth]   #permute转换维度  ，这里的深度等于 embeeding/nun_heads

    def forward(self, x, mask=False, multi_head=False):

        # batch_size seq_len 2      spatial_graph[8,57,2]

        assert len(x.shape) == 3

        embeddings = self.embedding(x)  # batch_size seq_len d_model
        query = self.query(embeddings)  # batch_size seq_len d_model
        key = self.key(embeddings)      # batch_size seq_len d_model

        if multi_head:
            query = self.split_heads(query)  # B num_heads seq_len d_model    #分别将Q和K分成多头，   [batch_size nun_heads seq_len depth] , depth=embeeding/num_head
            key = self.split_heads(key)  # B num_heads seq_len d_model
            attention = torch.matmul(query, key.permute(0, 1, 3, 2))  # (batch_size, num_heads, seq_len, seq_len)  是因为这里是Q*K的转置，所以最后两个调换了维度---[batch_size, num_heads,]
        else:                                                         #---[batch_size, num_heads,depth,seq_len] ---k是这个维度？？？（一样）不对，计算之后--最后应该是两个序列长度，因为呈完之后了
            attention = torch.matmul(query, key.permute(0, 2, 1))  # (batch_size, seq_len, seq_len)    这里也是，是因为这里是Q*K的转置，所以最后两个调换了维度，最后计算之后的维度为：[batch_size seq_len, seq_len]
                                                                     #[batch_size seq_len, embedding]----最后两个确实相等-----因为如果没有多头，Q和K的维度都是 [batch_size， seq_len embedding]
        attention = self.softmax(attention / self.scaled_factor)     # 多头之后维度为[batch_size，num_heads, seq_len, seq_len],不使用多头之后的维度：[batch_size,seq_len,seq_len]

        if mask is True:   #（传的参数是False）

            mask = torch.ones_like(attention)
            attention = attention * torch.tril(mask)  #将mask变成对角矩阵，右上面的值为0

        return attention, embeddings


class SpatialTemporalFusion(nn.Module):

    def __init__(self, obs_len=10):
        super(SpatialTemporalFusion, self).__init__()

        self.conv = nn.Sequential(
            nn.Conv2d(obs_len, obs_len, 1),
            nn.PReLU()
        )

        self.shortcut = nn.Sequential()

    def forward(self, x):

        x = self.conv(x) + self.shortcut(x)
        return x.squeeze()


class SparseWeightedAdjacency(nn.Module):   #得到邻接矩阵

    def __init__(self, spa_in_dims=4, tem_in_dims=5, embedding_dims=64, obs_len=10, dropout=0,
                 number_asymmetric_conv_layer=2):
        super(SparseWeightedAdjacency, self).__init__()
####这里obs_len换成12了
        # dense interaction
        self.spatial_attention = SelfAttention(spa_in_dims, embedding_dims)       #自注意力之后得到 attention和embedding
        self.temporal_attention = SelfAttention(tem_in_dims, embedding_dims)

        # 多头之后注意力维度为[batch_size，num_heads, seq_len, seq_len],
        # 不使用多头之后注意力的维度：[batch_size,seq_len,seq_len]
        # 嵌入向量的维度一直为：# batch_size seq_len d_model

        # attention fusion
        self.spa_fusion = SpatialTemporalFusion(obs_len=obs_len)    #卷积核1

        # interaction mask
        self.interaction_mask = InteractionMask(
            number_asymmetric_conv_layer=number_asymmetric_conv_layer
        )

        self.dropout = dropout
        self.zero_softmax = ZeroSoftmax()

    def forward(self, graph, identity):

        assert len(graph.shape) == 3   #graph 是三维的
        # # graph 1 obs_len N 3 ---------graph:[128,8,57,3]---[8,57,3]
        spatial_graph = graph[:, :, 1:]  # (T N 2)   (obs_len、行人数、坐标x和y)-----(batch_size、序列长度、嵌入维度)  ---这里T是obs_len吧？？？？确定了就是
        #所以空间图，最后是xy坐标  spatial_graph：graph:[128,8,57,2]---spatial_graph[8,57,2]
        temporal_graph = graph.permute(1, 0, 2)  # (N T 3)    (行人数、obs_len、坐标x和y 还有个（1，8+1）之间的随机值)   ---这里T是obs_len吧？？？？
        #时间图：temporal_graph:[128,8,57,3]--temporal_graph[8,57,3]
        # (T num_heads N N)   (T N d_model)  -------注意力之后变成（[batch_size，num_heads, seq_len, seq_len]）（ # 嵌入向量维度：batch_size seq_len d_model）,---T是batch_size
        dense_spatial_interaction, spatial_embeddings = self.spatial_attention(spatial_graph, multi_head=True) #-----[batch_size，num_heads, seq_len, seq_len]

        # (N num_heads T T)   (N T d_model)
        dense_temporal_interaction, temporal_embeddings = self.temporal_attention(temporal_graph, multi_head=True)

        # attention fusion   dense_spatial_interaction.permute(1, 0, 2, 3))=（num_heads,T,N,N）
        st_interaction = self.spa_fusion(dense_spatial_interaction.permute(1, 0, 2, 3)).permute(1, 0, 2, 3)     #(T num_heads N N) ）
        ts_interaction = dense_temporal_interaction

        spatial_mask, temporal_mask = self.interaction_mask(st_interaction, ts_interaction)  #这里st_interaction是融合了的，但是我没看出哪里表明融合了呢？
          #经过interaction_mask之后维度没有改变，spatial_mask:(T num_heads N N)  ;temporal_mask:(N num_heads T T)
        # self-connected
        spatial_mask = spatial_mask + identity[0].unsqueeze(1)     #[8,N,N]  N表示行人数，identity矩阵值是1，增加一维
        temporal_mask = temporal_mask + identity[1].unsqueeze(1)  # [N,8,8]

        normalized_spatial_adjacency_matrix = self.zero_softmax(dense_spatial_interaction * spatial_mask, dim=-1)   #dense_spatial_interaction就是注意力分数之后的矩阵
        normalized_temporal_adjacency_matrix = self.zero_softmax(dense_temporal_interaction * temporal_mask, dim=-1)

        return normalized_spatial_adjacency_matrix, normalized_temporal_adjacency_matrix,\
               spatial_embeddings, temporal_embeddings


class GraphConvolution(nn.Module):

    def __init__(self, in_dims=2, embedding_dims=16, dropout=0):
        super(GraphConvolution, self).__init__()

        self.embedding = nn.Linear(in_dims, embedding_dims, bias=False)
        self.activation = nn.PReLU()

        self.dropout = dropout

    def forward(self, graph, adjacency):

        # graph [batch_size 1 seq_len 2]
        # adjacency [batch_size num_heads seq_len seq_len]     邻接矩阵最后两个不应该是 行人数*行人数？？
        gcn_features = self.embedding(torch.matmul(adjacency, graph)) #torch.matmul用于不同维度的矩阵做乘法
        gcn_features = F.dropout(self.activation(gcn_features), p=self.dropout)

        return gcn_features  # [batch_size num_heads seq_len hidden_size]

def Mix(x,w1,b1,w2,b2):
    h = torch.tanh(torch.matmul(x,w1)+b1).cuda()
    f = torch.nn.Softmax(dim=-1)
    w = f(torch.matmul(h,w2)+b2).cuda()
    return w

class SparseGraphConvolution(nn.Module):

    def __init__(self, in_dims=16, embedding_dims=16, dropout=0):
        super(SparseGraphConvolution, self).__init__()

        self.dropout = dropout

        self.spatial_temporal_sparse_gcn = nn.ModuleList()
        self.temporal_spatial_sparse_gcn = nn.ModuleList()

        self.spatial_temporal_sparse_gcn.append(GraphConvolution(in_dims, embedding_dims))  #append()：在 ModuleList 后面添加网络层
        self.spatial_temporal_sparse_gcn.append(GraphConvolution(embedding_dims, embedding_dims))

        self.temporal_spatial_sparse_gcn.append(GraphConvolution(in_dims, embedding_dims))
        self.temporal_spatial_sparse_gcn.append(GraphConvolution(embedding_dims, embedding_dims))

        self.W1 = torch.tensor(np.random.normal(0, 0.01, (16, 64)), dtype=torch.float).cuda()
        self.b1 = torch.zeros(64, dtype=torch.float).cuda()
        self.W2 = torch.tensor(np.random.normal(0, 0.01, (64, 2)), dtype=torch.float).cuda()
        self.b2 = torch.zeros(2, dtype=torch.float).cuda()


    def forward(self, graph, normalized_spatial_adjacency_matrix, normalized_temporal_adjacency_matrix):

        # graph [1 seq_len num_pedestrians  3]     对的，这里1表示batch_size
        # _matrix [batch num_heads seq_len seq_len]   对的

        graph = graph[:, :, :, 1:]
        spa_graph = graph.permute(1, 0, 2, 3)  # (seq_len 1 num_p 2)
        tem_graph = spa_graph.permute(2, 1, 0, 3)  # (num_p 1 seq_len 2)

        gcn_spatial_features0 = self.spatial_temporal_sparse_gcn[0](spa_graph, normalized_spatial_adjacency_matrix)  # [batch_size num_heads seq_len hidden_size]
        gcn_spatial_features = gcn_spatial_features0.permute(2, 1, 0, 3)  #(8,4,N,16)->(N,4,8,16)

        # [num_p num_heads seq_len d]
        # print("****-----")
        # print(gcn_spatial_features.shape)
        # print('tem-adj---',normalized_temporal_adjacency_matrix.shape)
        # print(self.spatial_temporal_sparse_gcn[1])
        gcn_spatial_temporal_features = self.spatial_temporal_sparse_gcn[1](gcn_spatial_features, normalized_temporal_adjacency_matrix)
        # print('gcn-st---,',gcn_spatial_temporal_features.shape)

        gcn_temporal_features0 = self.temporal_spatial_sparse_gcn[0](tem_graph,normalized_temporal_adjacency_matrix)
        gcn_temporal_features1 = gcn_temporal_features0.permute(2, 1, 0, 3)
        gcn_temporal_spatial_features = self.temporal_spatial_sparse_gcn[1](gcn_temporal_features1,normalized_spatial_adjacency_matrix)

        x = gcn_spatial_features0 + gcn_temporal_features1   #(8,4,n,16)
        # w = Mix(x,self.W1,self.b1,self.W2,self.b2)
        # w1 = w[:,:,:,0]   #(8,4,N)
        # w2 = w[:,:,:,1]
        # gcn_spatial_features0 = gcn_spatial_features0.permute(3,0,1,2)
        # gcn_temporal_features1 = gcn_temporal_features1.permute(3,0,1,2)
        # H = w1*gcn_spatial_features0 + w2*gcn_temporal_features1   #(16,8,4,n)
        # H = H.permute(3,1,2,0)
        H = x.permute(2,0,1,3)

        # gcn_spatial_temporal_features, gcn_temporal_spatial_features.permute(2, 1, 0, 3)
        return H

class TCN(nn.Module):
    def __init__(self,fin,fout,layers=3,ksize=3):
        super(TCN, self).__init__()
        self.fin = fin
        self.fout = fout
        self.layers = layers
        self.ksize = ksize

        self.convs = nn.ModuleList()
        for i in range(self.layers):
            self.convs.append(nn.Conv2d(self.fin,self.fout,kernel_size=self.ksize))

    def forward(self,x):
        for _,conv_layer in enumerate(self.convs):
            x = nn.functional.pad(x,(self.ksize-1,0,self.ksize-1,0))
            x = conv_layer(x)
        return x



class Encoder(nn.Module):
    def __init__(self,fin,fout,layers=3,ksize=3):
        super(Encoder, self).__init__()
        self.fin = fin
        self.fout = fout
        self.layers = layers
        self.ksize = ksize

        self.tcnf = TCN(self.fin, self.fout, self.layers, self.ksize)
        self.tcng = TCN(self.fin, self.fout, self.layers, self.ksize)

    def forward(self,x):
        residual = x
        f = torch.sigmoid(self.tcnf(x))
        g = torch.tanh(self.tcng(x))
        return residual + f*g


class TrajectoryModel(nn.Module):

    def __init__(self,
                 number_asymmetric_conv_layer=2, embedding_dims=64, number_gcn_layers=1, dropout=0,
                 obs_len=8, pred_len=12,
                 out_dims=5, num_heads=4):
        super(TrajectoryModel, self).__init__()

        self.number_gcn_layers = number_gcn_layers
        self.obs_len = obs_len
        self.pred_len = pred_len
        self.dropout = dropout

        # sparse graph learning
        self.sparse_weighted_adjacency_matrices = SparseWeightedAdjacency(
            number_asymmetric_conv_layer=number_asymmetric_conv_layer,obs_len=obs_len
        )

        # graph convolution
        self.stsgcn = SparseGraphConvolution(
            in_dims=4, embedding_dims=embedding_dims // num_heads, dropout=dropout
        )

        self.fusion_ = nn.Conv2d(num_heads, num_heads, kernel_size=1, bias=False)

        # self.tcns = nn.ModuleList()
        # self.tcns.append(nn.Sequential(
        #     nn.Conv2d(obs_len, pred_len, 3, padding=1),
        #     nn.PReLU()
        # ))
        #
        # for j in range(1, self.n_tcn):  #1,2,3,4
        #     self.tcns.append(nn.Sequential(
        #         nn.Conv2d(pred_len, pred_len, 3, padding=1),
        #         nn.PReLU()
        # ))
        self.encoder = Encoder(fin=self.obs_len,fout=pred_len)

        # self.output = nn.Linear(embedding_dims // num_heads, out_dims)
        self.output = nn.Linear(embedding_dims , out_dims)
        # self.tcnf = nn.Conv2d(obs_len, pred_len, 3, padding=1)
        # self.tcng = nn.Conv2d(obs_len, pred_len, 3, padding=1)


    def forward(self, graph, identity):
        # V_obs:[128,8,57,3]  -----空间邻接矩阵identity_spatial：[8,57,57]---  ---时间邻接矩阵identity_temporal：[57,8,8]
        # graph 1 obs_len N 3 -----这个对，就是V_obs，batch_size,obs_len,N(行人数)，3

        normalized_spatial_adjacency_matrix, normalized_temporal_adjacency_matrix, spatial_embeddings, temporal_embeddings = \
            self.sparse_weighted_adjacency_matrices(graph.squeeze(), identity)   #得到归一化之后的离散空间，时间邻接矩阵

        H = self.stsgcn(
            graph, normalized_spatial_adjacency_matrix, normalized_temporal_adjacency_matrix
        )   #通过两个GCN得到 spatial-temporal 和temporal-spatial   两个图卷积特征，我不理解的点是 gcn中哪里体现出了？我只看到了线性加上激活函数，是因为输出已经是graph了？？
#序列长度，多头，batch_size,hidden_size
        # gcn_representation = self.fusion_(gcn_temporal_spatial_features) + gcn_spatial_temporal_features
        # gcn_representation = self.fusion_(gcn_temporal_spatial_features) + self.fusion_(gcn_spatial_temporal_features)
        # gcn_representation = gcn_temporal_spatial_features + gcn_spatial_temporal_features  #[N,4,obs,16]

        gcn_representation = H

        # features = self.tcns[0](gcn_representation)   #[N,12,4,16]

        features = self.encoder(gcn_representation)
        b,l,_,_ = features.shape
        features = features.contiguous().view(b,self.pred_len,-1)
        prediction = self.output(features)

        # f = torch.sigmoid(self.tcnf(gcn_representation))
        # g = torch.tanh(self.tcng(gcn_representation))
        # features = gcn_representation + f * g


        # prediction = torch.mean(self.output(features), dim=-2)
        #
        # f = torch.sigmoid(self.tcnf(prediction))
        # g = torch.tanh(self.tcng(prediction))
        # prediction = prediction + f * g

        #
        # for k in range(1, self.n_tcn):  #1,2,3,4
        #     features = F.dropout(self.tcns[k](features) + features, p=self.dropout)

        # prediction = torch.mean(self.output(features), dim=-2)   #dim=-2相当于dim=0,按列求平均值，有几列就输出几列

        return prediction.permute(1, 0, 2).contiguous()
