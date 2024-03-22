import random 
import os 
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch.utils.data import Dataset, DataLoader
import pickle
from transformers import BertTokenizer, BertModel
import pandas as pd
from queue import Queue
from tqdm import tqdm
import dgl
from log_model import Logger
import argparse
from dgl.nn.pytorch import GraphConv, GATConv, SAGEConv
from dgl.nn import GlobalAttentionPooling
from geomloss import SamplesLoss
from collections import Counter
#加载日志的类
logger = Logger('logs/',level='debug')


parser = argparse.ArgumentParser()
parser.add_argument('--batch_size', type=int, default=32)
parser.add_argument('--epochs', type=int, default=100)
parser.add_argument('--device', type=str, default="0")
parser.add_argument('--hidden_size', type=int, default=256)
parser.add_argument('--num_layers', type=int, default=6)
parser.add_argument('--show_step', type=int, default=20)
parser.add_argument('--lr', type=float, default=0.001)
parser.add_argument('--min_lr', type=float, default=0.0001)
parser.add_argument('--dataset', type=str, default=None)
parser.add_argument('--save_step', type=int, default=10000)

args = parser.parse_args()

device = "cuda:" + args.device

logger.logger.info(str(args))


class gcn_layer(nn.Module):
    def __init__(self, nfeat, nhid):
        super(gcn_layer, self).__init__()
        self.conv = GraphConv(nfeat, nhid,norm="both",allow_zero_in_degree=True)
        self.ly = nn.LayerNorm(nhid)
        
    def forward(self, g, h):
        x = self.conv(g, h)
        x = self.ly(x)
        return x


class gsage_layer(nn.Module):
    def __init__(self, nfeat, nhid):
        super(gsage_layer, self).__init__()
        self.conv = SAGEConv(nfeat, nhid, 'pool')
        self.ly = nn.LayerNorm(nhid)
        
    def forward(self, g, h):
        x = self.conv(g, h) + h
        x = self.ly(x)
        return x



class Encoder(nn.Module):
    def __init__(self, hidden_size, dropout, num_layers):
        super(Encoder, self).__init__()
        self.convs = torch.nn.ModuleList()
        for _ in range(num_layers):
            x_size = 2*hidden_size
            y_size = 2*hidden_size
            self.convs.append(gsage_layer(x_size, y_size))   
        self.dropout = dropout 
    def forward(self, g, h):
        num_layers = len(self.convs)
        for i, conv in enumerate(self.convs):
            h = conv(g, h)
            if i < num_layers - 1:
                h = h.relu_()
                h = F.dropout(h, p=self.dropout, training=self.training)
        return h


class PreModel(nn.Module):
    def __init__(self, hidden_size, dropout, num_layers, out_size):
        super(PreModel, self).__init__()
        self.in_linear = nn.ModuleDict({"tsocial":nn.Linear(10, hidden_size),
                                        "dgraphfin":nn.Linear(17, hidden_size),
                                        "tfinance":nn.Linear(10, hidden_size)})
        self.GCN = gcn_layer(hidden_size, hidden_size)
        gate_nn = torch.nn.Linear(hidden_size, 1)
        self.gap_pool = GlobalAttentionPooling(gate_nn)
        self.encoder = Encoder(hidden_size, dropout, num_layers)
        self.out_linear = nn.Linear(2*hidden_size, out_size)

        self.trans_linear_1 = nn.Linear(64, 64)
        self.LeakyReLU = nn.LeakyReLU(negative_slope=5e-2)
        self.trans_linear_2 = nn.Linear(64, 64)

    def forward(self, g_name, g, h):
        h = self.in_linear[g_name](h)

        h = self.trans_linear_1(h)
        h = self.LeakyReLU(h)
        h = self.trans_linear_2(h)


        #邻居减法
        loc_h = self.GCN(g, h) - h
        #全局减法
        pool_h = self.gap_pool(g, h)
        glo_h = pool_h.expand(h.shape[0], -1) - h

        Z = torch.cat([loc_h, glo_h], dim=1)

        h = self.encoder(g, Z)
        score = self.out_linear(h)
        return score, Z



def create_data_loader(graph):
    sampler = dgl.dataloading.NeighborSampler([128, 32])
    labels = graph.ndata["label"].numpy()
    train_idx = list(np.where(labels==0)[0]) + list(np.where(labels==1)[0])
    train_idx = np.array(train_idx)

    train_loader = dgl.dataloading.DataLoader(
            graph, train_idx, sampler,
            batch_size=args.batch_size,
            shuffle=True,
            num_workers=8)
    return train_loader


def load_graph(file_path):
    graph = dgl.load_graphs(file_path)[0][0]
    logger.logger.info("loading graph from {} feat_size={}".format(file_path, graph.ndata["feat"].shape))
    train_loader = create_data_loader(graph)
    return graph, train_loader


def create_sub_graph(batch_data):

    input_nodes, output_nodes, blocks = batch_data
    input_nodes = input_nodes.cpu().numpy()

    edges = blocks[0].edges()
    us = edges[0]
    vs = edges[1]
    #添加反向边
    new_us = torch.cat([us, vs], dim=0)
    new_vs = torch.cat([vs, us], dim=0)
    #创建子图
    sub_graph = dgl.graph((new_us, new_vs)).to(device)

    input_features = blocks[0].srcdata['feat'].to(device)
    
    cur_bz = len(output_nodes)
    labels = blocks[-1].dstdata["label"].to(device)

    return sub_graph, input_features, cur_bz, labels


def dev_loss(y_pred, y_true):
    confidence_margin = 5.0
    # size=5000 is the setting of l in algorithm 1 in the paper
    ref = torch.normal(mean=0., std=torch.full([5000], 1.0)).to(device)
    dev = (y_pred - torch.mean(ref)) / torch.std(ref)
    inlier_loss = torch.abs(dev)
    outlier_loss = torch.abs((confidence_margin - dev).clamp_(min=0.))
    return torch.mean((1 - y_true) * inlier_loss + y_true * outlier_loss)



if __name__ == "__main__":
    file_name_ls = ["tsocial", "dgraphfin", "tfinance"]
    #file_name_ls = ["wiki", "reddit", "elliptic"]

    graph_loader_dic = {}
    for file_name in file_name_ls:
        graph, train_loader = load_graph("../PretrainGNN/datasets/{}".format(file_name))
        graph_loader_dic[file_name] = {}
        graph_loader_dic[file_name]["train_loader"] = train_loader
    

    model = PreModel(hidden_size=args.hidden_size, 
                out_size=2,
                dropout=0.3, 
                num_layers=5).to(device)

    #----------------下面是模型的参数训练的一个调整----------
    param_optimizer = list(model.named_parameters())#得到模型的参数
    #----------------下面是模型的参数训练的一个调整----------
    no_decay = ['bias', 'gamma', 'beta']
    optimizer_grouped_parameters = [
        {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay) and "trans" not in n],
        'weight_decay_rate': 0.01},
        {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay) and "trans" not in n],
        'weight_decay_rate': 0.0},
        {'params': [p for n, p in param_optimizer if "trans" in n],
        'weight_decay_rate': 0.0, 'lr':0.005}
    ]
    '''
    optimizer_grouped_parameters = [
        {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)],
        'weight_decay_rate': 0.01},
        {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)],
        'weight_decay_rate': 0.0},
    ]
    '''
    
    #这里是训练的优化器选择，Adam，学习率是2e-5
    opt = torch.optim.AdamW(optimizer_grouped_parameters, 
                            lr=args.lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(opt, T_0=50, T_mult=2, eta_min=args.min_lr, last_epoch=-1)

    sh_p = 2
    sh_scaling = 0.75
    sh_blur = 0.05
    sinkhorn = SamplesLoss(loss='sinkhorn',
                                p=sh_p,
                                scaling=sh_scaling,
                                blur=sh_blur)
    loss_all = []
    step_num = 0
    for epoch_id in range(args.epochs):
        for file_name_1 in file_name_ls:
            for file_name_2 in file_name_ls:
                if file_name_1 == file_name_2:
                    continue
                train_loader_1 = graph_loader_dic[file_name_1]["train_loader"]
                train_loader_2 = graph_loader_dic[file_name_2]["train_loader"]
                total_batch_num = min(len(train_loader_1), len(train_loader_2))
                for batch_idx, batch_data in enumerate(zip(train_loader_1, train_loader_2)):
                    batch_data_1, batch_data_2 = batch_data
        
                    sub_graph_1, feats_1, cur_bz_1, labels_1 = create_sub_graph(batch_data_1)
                    sub_graph_2, feats_2, cur_bz_2, labels_2 = create_sub_graph(batch_data_2)

                    #如果两个批次大小不一致，结束训练
                    if cur_bz_1 != cur_bz_2:
                        break

                    score_1, Z_1 = model(file_name_1, sub_graph_1, feats_1)
                    score_2, Z_2 = model(file_name_2, sub_graph_2, feats_2)
                    
                    score_1 = score_1[0:cur_bz_1]
                    score_2 = score_2[0:cur_bz_2]

                    Z_1 = Z_1[0:cur_bz_1]
                    Z_2 = Z_2[0:cur_bz_2]

                    score_1 = F.softmax(score_1, dim=1)
                    score_2 = F.softmax(score_2, dim=1)

                    score = torch.cat([score_1, score_2], dim=0)
                    labels = torch.cat([labels_1, labels_2], dim=0)

                    loss_dev = dev_loss(score[:,1], labels)
                    loss_align = sinkhorn(Z_1, Z_2)/128

                    loss = loss_align + loss_dev

                    opt.zero_grad()
                    loss.backward()
                    opt.step()
                    scheduler.step()
                    loss_all.append(loss.item())
                    step_num += 1
                    if step_num % args.show_step == 0:
                        logger.logger.info("epoch={}/{} batch={}/{} lr={} loss={}".format(epoch_id, args.epochs, batch_idx, total_batch_num, opt.param_groups[-1]['lr'], np.mean(loss_all)))
                        loss_all = []
                    if step_num % args.save_step == 0:
                        save_path = "models/pretrain_encoder_{}.pt".format(step_num)
                        logger.logger.info("save model path={}".format(save_path))
                        torch.save(model.state_dict(), save_path)


            

        






'''
datasets/yelp
datasets/tsocial
python run_pretrain.py\
        --batch_size 256\
        --device 1\
        --lr 1e-3\
        --min_lr 2e-4\
        --hidden_size 64\
        --save_step 3000

'''

