import random 
import os 
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import dgl
from dgl.nn.pytorch import GraphConv, GATConv, SAGEConv
from collections import Counter
from log_model import Logger
from dgl.nn import GlobalAttentionPooling
from torch.utils.data import Dataset, DataLoader
from dgl.dataloading import GraphDataLoader
import dgl
from sklearn.metrics import average_precision_score, f1_score, roc_auc_score
from dgl.transforms import to_block
from dgl.base import NID, EID
from dgl.nn.pytorch import AvgPooling, Set2Set
import argparse
from tqdm import tqdm
#torch.nn.init.kaiming_uniform_(tensor, a=0, mode='fan_in', nonlinearity='leaky_relu')

#加载日志的类
logger = Logger('logs/',level='debug')


parser = argparse.ArgumentParser()
parser.add_argument('--dataset', type=str, default='wiki')
parser.add_argument('--batch_size', type=int, default=32)
parser.add_argument('--epochs', type=int, default=100)
parser.add_argument('--device', type=str, default="0")
parser.add_argument('--num_prompt', type=int, default=1)
parser.add_argument('--prompt_size', type=int, default=64)
parser.add_argument('--no_up', type=int, default=100)
parser.add_argument('--hidden_size', type=int, default=128)
parser.add_argument('--num_layers', type=int, default=2)
parser.add_argument('--show_step', type=int, default=20)
parser.add_argument('--base_model', type=str, default="GCN")
parser.add_argument('--lr', type=float, default=0.001)
parser.add_argument('--min_lr', type=float, default=0.0001)
parser.add_argument('--pretrain_model', type=str, default=None)


args = parser.parse_args()

device = "cuda:" + args.device

logger.logger.info(str(args))


#定义一个数据集
class MyDataset(Dataset):
    def __init__(self, nids):
        self.nids = nids
        self.length = len(nids)

    def __len__(self):
        return self.length

    def __getitem__(self, index):
        src_id = self.nids[index]
        return src_id




class gin_layer(nn.Module):
    def __init__(self, nfeat, nhid):
        super(gin_layer, self).__init__()
        lin = nn.Linear(nfeat, nhid)
        self.conv = GINConv(lin, 'max', activation=relu)
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
        x = self.conv(g, h)
        x = self.ly(x)
        return x


class gcn_layer(nn.Module):
    def __init__(self, nfeat, nhid):
        super(gcn_layer, self).__init__()
        self.conv = GraphConv(nfeat, nhid,norm="both",allow_zero_in_degree=True)
        self.ly = nn.LayerNorm(nhid)
        
    def forward(self, g, h):
        x = self.conv(g, h)
        x = self.ly(x)
        return x


class gat_layer(nn.Module):
    def __init__(self, nfeat, nhid):
        super(gat_layer, self).__init__()
        self.conv = GATConv(nfeat, nhid//2, num_heads = 2,allow_zero_in_degree=True)
        self.ly = nn.LayerNorm(nhid)
        
    def forward(self, g, h):
        x = self.conv(g, h)
        shape = x.shape
        x = x.reshape(shape[0], -1)
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
        for i, conv in enumerate(self.convs):
            h = conv(g, h)
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



class Model(nn.Module):
    def __init__(self, in_size, hidden_size, dropout, num_layers, out_size):
        super(Model, self).__init__()
        self.trans_in_linear = nn.Linear(in_size, hidden_size)
        self.GCN = gcn_layer(hidden_size, hidden_size)
        gate_nn = torch.nn.Linear(hidden_size, 1)
        self.gap_pool = GlobalAttentionPooling(gate_nn)
        self.encoder = Encoder(hidden_size, dropout, num_layers)
        self.out_linear = nn.Linear(2*hidden_size, out_size)

        self.trans_linear_1 = nn.Linear(64, 64)
        self.LeakyReLU = nn.LeakyReLU(negative_slope=5e-2)
        self.trans_linear_2 = nn.Linear(64, 64)

    def forward(self, g, h):
        h = self.trans_in_linear(h)
        
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
        return score


def cal_auc_ap_f1score(preds, labels):
    scores = preds[:,1]
    pred_labels = preds.argmax(axis=1)
    auc = roc_auc_score(labels, scores)
    ap = average_precision_score(labels, scores)
    f1 = f1_score(labels, pred_labels, average='macro')

    return auc, ap, f1

def dev_loss(y_pred, y_true):
    confidence_margin = 5.0
    # size=5000 is the setting of l in algorithm 1 in the paper
    ref = torch.normal(mean=0., std=torch.full([5000], 1.0)).to(device)
    dev = (y_pred - torch.mean(ref)) / torch.std(ref)
    inlier_loss = torch.abs(dev)
    outlier_loss = torch.abs((confidence_margin - dev).clamp_(min=0.))
    return torch.mean((1 - y_true) * inlier_loss + y_true * outlier_loss)

def calculate():
    logger.logger.info("loading dataset_name="+args.dataset)
    #graph = load_truth_data(data_path="dataset",dataset_name=args.dataset)
    graph = dgl.load_graphs(args.dataset)[0][0]

    logger.logger.info(str(graph))

    
    labels = graph.ndata['label'].numpy()
    feats = graph.ndata['feat'].numpy()
    train_mask = graph.ndata['train_mask'].numpy()
    val_mask = graph.ndata['val_mask'].numpy()
    test_mask = graph.ndata['test_mask'].numpy()

    new_labels = []
    for nid in np.arange(feats.shape[0]):
        if train_mask[nid] == True or val_mask[nid] == True:
            new_labels.append(labels[nid])
        else:
            new_labels.append(2)
    
    new_labels = np.array(new_labels)

    train_nids = np.argwhere(graph.ndata['train_mask'].numpy() == True).reshape(-1)
    val_nids = np.argwhere(graph.ndata['val_mask'].numpy() == True).reshape(-1)
    test_nids = np.argwhere(graph.ndata['test_mask'].numpy() == True).reshape(-1)

    logger.logger.info("train:val:test={}:{}:{}".format(len(train_nids), len(val_nids), len(test_nids)))

    model = Model(in_size=graph.ndata['feat'].shape[1], 
                hidden_size=args.hidden_size, 
                out_size=2, 
                dropout=0.3, 
                num_layers=args.num_layers).to(device)
    
    if args.pretrain_model != None:
        premodel = PreModel(hidden_size=args.hidden_size, 
                out_size=2,
                dropout=0.3, 
                num_layers=5).to(device)
        logger.logger.info("loading pretrain_model from {}".format(args.pretrain_model))
        premodel.load_state_dict(torch.load(args.pretrain_model))
        model.encoder = premodel.encoder
        model.trans_linear_1 = premodel.trans_linear_1
        model.LeakyReLU = premodel.LeakyReLU
        model.trans_linear_2 = premodel.trans_linear_2
        model.GCN = premodel.GCN
        model.gap_pool = premodel.gap_pool
        #model.linear = premodel.linear
        del premodel
    
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
    #这里是训练的优化器选择，AdamW，学习率是2e-5
    opt = torch.optim.AdamW(optimizer_grouped_parameters, 
                            lr=args.lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(opt, T_0=50, T_mult=2, eta_min=args.min_lr, last_epoch=-1)
    loss_function = nn.CrossEntropyLoss()
    best_val_score = 0
    test_auc = 0
    test_ap = 0
    test_f1 = 0
    loss_all = []

    train_mask = graph.ndata['train_mask'].bool()
    val_mask = graph.ndata['val_mask'].bool()
    test_mask = graph.ndata['test_mask'].bool()
    labels = graph.ndata['label'].long().to(device)
    
    step_num = 0
    no_up_num = 0
    for epoch_id in range(args.epochs):
        out = model(graph.to(device), graph.ndata['feat'].to(device))
        loss = loss_function(out[train_mask], labels[train_mask])
        opt.zero_grad()
        loss.backward()
        opt.step()
        scheduler.step()
        loss_all.append(loss.item())
        step_num += 1
        if step_num % args.show_step == 0:
            logger.logger.info("epoch={}/{} lr={} loss={}".format(epoch_id, args.epochs, opt.param_groups[-1]['lr'], np.mean(loss_all)))
            loss_all = []
        
        out = F.softmax(out, dim=1)
        
        #验证集
        val_preds = out[val_mask].detach().cpu().numpy()
        val_labels = labels[val_mask].cpu().numpy()
        val_auc, val_ap, val_f1 = cal_auc_ap_f1score(val_preds, val_labels)
        if val_auc > best_val_score:
            logger.logger.info("epoch={}/{} val: auc={} ap={} f1={}".format(epoch_id, args.epochs, val_auc, val_ap, val_f1))
            best_val_score = val_auc
            #测试集
            test_preds = out[test_mask].detach().cpu().numpy()
            test_labels = labels[test_mask].cpu().numpy()
            test_auc, test_ap, test_f1 = cal_auc_ap_f1score(test_preds, test_labels)
            logger.logger.info("epoch={}/{} best_val_auc={} ===> best_test: auc={} ap={} f1={}".format(epoch_id, args.epochs, best_val_score, test_auc, test_ap, test_f1))
            no_up_num = 0
        else:
            no_up_num += 1
        if no_up_num >= args.no_up:
            logger.logger.info("The val score is not up for {} epochs".format(args.no_up))
            break
    
    logger.logger.info("final best_val_auc={} ===> best_test: auc={} ap={} f1={}".format(best_val_score, test_auc, test_ap, test_f1))
    return test_auc, test_ap, test_f1




if __name__ == "__main__":
    test_auc_ls = []
    test_ap_ls = []
    test_f1_ls = []
    for _ in range(10):
        test_auc, test_ap, test_f1 = calculate()
        test_auc_ls.append(test_auc)
        test_ap_ls.append(test_ap)
        test_f1_ls.append(test_f1)
    
    logger.logger.info("auc : mean="+str(np.mean(test_auc_ls)*100)+" std="+str(np.std(test_auc_ls)*100))
    logger.logger.info("ap : mean="+str(np.mean(test_ap_ls)*100)+" std="+str(np.std(test_ap_ls)*100))
    logger.logger.info("f1 : mean="+str(np.mean(test_f1_ls)*100)+" std="+str(np.std(test_f1_ls)*100))



'''
增加了邻居减法特征和全局减法特征显性提示

tsocial
dgraphfin
tfinance



tfinance
reddit
wiki
amazon
yelp
elliptic
mimic
alpha

python run_finetune.py --dataset ../datasets/yelp/homo.dgl\
                        --batch_size 256\
                        --device 5\
                        --show_step 20\
                        --epochs 2000\
                        --no_up 1000\
                        --base_model GraphSage\
                        --hidden_size 64\
                        --lr 1e-3\
                        --min_lr 2e-4\
                        --num_layers 5\
                        --pretrain_model models/pretrain_encoder_60000.pt


/data1/fangmengcheng/GraphAnomalyDetection/CARE-GNN-masterGCN
GAT
GraphSage
GIN

'''