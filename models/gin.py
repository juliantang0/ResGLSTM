# -*- coding = utf-8 -*-
# @Time : 2023/4/15 14:38
# @Author : 汤羽舟
# @File : gin.py
# @Software : PyCharm

import torch
import torch.nn as nn
from torch.nn import Sequential, Linear, ReLU
import torch.nn.functional as F
from torch_geometric.nn import GINConv, global_add_pool


class ResGINNet(torch.nn.Module):
    def __init__(self, k1, k2, k3, embed_dim, num_layer, device, num_feature_xd=78, n_output=1, num_feature_xt=25,
                 output_dim=128, dropout=0.2):
        super(ResGINNet, self).__init__()
        self.device = device
        # Smile graph branch
        self.k1 = k1
        self.k2 = k2
        self.k3 = k3
        self.embed_dim = embed_dim
        self.num_layer = num_layer
        dim = 32
        self.Conv1 = GINConv(Sequential(Linear(num_feature_xd, dim), ReLU(), Linear(dim, dim)))
        self.bn1 = torch.nn.BatchNorm1d(dim)
        self.Conv2 = GINConv(Sequential(Linear(dim, dim), ReLU(), Linear(dim, dim)))
        self.bn2 = torch.nn.BatchNorm1d(dim)
        self.Conv3 = GINConv(Sequential(Linear(dim, dim), ReLU(), Linear(dim, dim)))
        self.bn3 = torch.nn.BatchNorm1d(dim)
        self.Conv4 = GINConv(Sequential(Linear(dim, dim), ReLU(), Linear(dim, dim)))
        self.bn4 = torch.nn.BatchNorm1d(dim)
        self.Conv5 = GINConv(Sequential(Linear(dim, dim), ReLU(), Linear(dim, dim)))
        self.bn5 = torch.nn.BatchNorm1d(dim)
        self.relu = nn.ReLU()
        self.fc_g1 = Linear(dim * 5, output_dim)
        self.dropout = nn.Dropout(dropout)

        # protien sequence branch (LSTM)
        self.embedding_xt = nn.Embedding(num_feature_xt + 1, embed_dim)
        self.LSTM_xt_1 = nn.LSTM(self.embed_dim, self.embed_dim, self.num_layer, batch_first=True, bidirectional=True)
        self.fc_xt = nn.Linear(1000 * 256, output_dim)

        # combined layers
        self.fc1 = nn.Linear(2 * output_dim, 1024)
        self.fc2 = nn.Linear(1024, 512)
        self.out = nn.Linear(512, n_output)

    def forward(self, data, hidden, cell):
        x, edge_index, batch = data.x, data.edge_index, data.batch  # 固定写法，用来训练网络
        target = data.target

        h1 = F.relu(self.Conv1(x, edge_index))
        h1 = self.bn1(h1)

        h2 = F.relu(self.Conv2(h1, edge_index))
        h2 = self.bn2(h2)

        h3 = F.relu(self.Conv3(h2, edge_index))
        h3 = self.bn3(h3)

        h4 = F.relu(self.Conv4(h3, edge_index))
        h4 = self.bn4(h4)

        h5 = F.relu(self.Conv5(h4, edge_index))
        h5 = self.bn5(h5)

        h1 = global_add_pool(h1, batch)
        h2 = global_add_pool(h2, batch)
        h3 = global_add_pool(h3, batch)
        h4 = global_add_pool(h4, batch)
        h5 = global_add_pool(h5, batch)

        x = torch.cat([h1, h2, h3, h4, h5], dim=1)

        # flatten
        x = F.relu(self.fc_g1(x))
        x = self.dropout(x)

        # LSTM layer
        embedded_xt = self.embedding_xt(target)
        LSTM_xt, (hidden, cell) = self.LSTM_xt_1(embedded_xt, (hidden, cell))
        xt = LSTM_xt.contiguous().view(-1, 1000 * 256)
        xt = self.fc_xt(xt)

        # concat
        xc = torch.cat((x, xt), 1)
        # add some dense layers
        xc = self.fc1(xc)
        xc = self.relu(xc)
        xc = self.dropout(xc)
        xc = self.fc2(xc)
        xc = self.relu(xc)
        xc = self.dropout(xc)
        out = self.out(xc)
        return out

    def init_hidden(self, batch_size):
        hidden = torch.zeros(2 * self.num_layer, batch_size, self.embed_dim).to(self.device)
        cell = torch.zeros(2 * self.num_layer, batch_size, self.embed_dim).to(self.device)
        return hidden, cell
