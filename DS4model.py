#!/usr/bin/env python
# -*- coding: utf-8 -*-
from __future__ import absolute_import
from __future__ import print_function

import torch.nn as nn
import torch
from torch.nn.parameter import Parameter
import math
import numpy as np


# GCN implementation comes from https://github.com/wei-mao-2019/LearnTrajDep
class GraphConvolution(nn.Module):
    """
    adapted from : https://github.com/tkipf/gcn/blob/92600c39797c2bfb61a508e52b88fb554df30177/gcn/layers.py#L132
    """

    def __init__(self, in_features, out_features, bias=True, node_n=48):
        super(GraphConvolution, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = Parameter(torch.FloatTensor(in_features, out_features))
        # self.att = Parameter(torch.FloatTensor(node_n, node_n))
        self.att = Parameter(torch.FloatTensor(0.01 + 0.99 * np.eye(node_n)[np.newaxis, ...]))
        if bias:
            self.bias = Parameter(torch.FloatTensor(out_features))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self):
        # stdv = 1. / math.sqrt(self.weight.size(1))
        # self.weight.data.uniform_(-stdv, stdv)
        # self.att.data.uniform_(-stdv, stdv)
        torch.nn.init.xavier_normal(self.weight)
        if self.bias is not None:
            # self.bias.data.uniform_(-stdv, stdv)
            self.bias.data.zero_()

    def forward(self, input):
        support = torch.matmul(input, self.weight)
        output = torch.matmul(self.att, support)
        if self.bias is not None:
            return output + self.bias
        else:
            return output

    def __repr__(self):
        return self.__class__.__name__ + ' (' \
               + str(self.in_features) + ' -> ' \
               + str(self.out_features) + ')'

class GC_Block(nn.Module):
    def __init__(self, in_features, p_dropout, bias=True, node_n=48):
        """
        Define a residual block of GCN
        """
        super(GC_Block, self).__init__()
        self.in_features = in_features
        self.out_features = in_features

        self.gc1 = GraphConvolution(in_features, in_features, node_n=node_n, bias=bias)
        self.bn1 = nn.BatchNorm1d(node_n * in_features)

        self.gc2 = GraphConvolution(in_features, in_features, node_n=node_n, bias=bias)
        self.bn2 = nn.BatchNorm1d(node_n * in_features)

        self.do = nn.Dropout(p_dropout)
        self.act_f = nn.Tanh()

    def forward(self, x):
        y = self.gc1(x)
        b, n, f = y.shape
        y = self.bn1(y.view(b, -1)).view(b, n, f)
        y = self.act_f(y)
        y = self.do(y)

        y = self.gc2(y)
        b, n, f = y.shape
        y = self.bn2(y.view(b, -1)).view(b, n, f)
        y = self.act_f(y)
        y = self.do(y)

        return y + x



    def __repr__(self):
        return self.__class__.__name__ + ' (' \
               + str(self.in_features) + ' -> ' \
               + str(self.out_features) + ')'

class GC2(nn.Module):
    def __init__(self, in_features, p_dropout, bias=True, node_n=48):
        """
        Define a residual block of GCN
        """
        super(GC2, self).__init__()
        self.in_features = in_features
        self.out_features = in_features

        self.gc1 = GraphConvolution(in_features, in_features, node_n=node_n, bias=bias)
        self.bn1 = nn.BatchNorm1d(node_n * in_features)

        self.gc2 = GraphConvolution(in_features, in_features, node_n=node_n, bias=bias)
        self.bn2 = nn.BatchNorm1d(node_n * in_features)

        self.gc3 = GraphConvolution(in_features, in_features, node_n=node_n, bias=bias)
        self.bn3 = nn.BatchNorm1d(node_n * in_features)

        self.gc4 = GraphConvolution(in_features, in_features, node_n=node_n, bias=bias)
        self.bn4 = nn.BatchNorm1d(node_n * in_features)

        self.gc5 = GraphConvolution(in_features, in_features, node_n=node_n, bias=bias)
        self.bn5 = nn.BatchNorm1d(node_n * in_features)

        self.do = nn.Dropout(p_dropout)
        self.act_f = nn.Tanh()

    def forward(self, x):
        y1 = self.gc1(x)
        b, n, f = y1.shape
        y1 = self.bn1(y1.view(b, -1)).view(b, n, f)
        y1 = self.act_f(y1)
        y1 = self.do(y1)

        y2 = self.gc2(y1)
        b, n, f = y2.shape
        y2 = self.bn2(y2.view(b, -1)).view(b, n, f)
        y2 = self.act_f(y2)
        y2 = self.do(y2)

        y3 = self.gc3(y2)
        b, n, f = y3.shape
        y3 = self.bn3(y3.view(b, -1)).view(b, n, f)
        y3 = self.act_f(y3)
        y3 = self.do(y3)

        y4 = self.gc4(y3 + y2)
        b, n, f = y4.shape
        y4 = self.bn4(y4.view(b, -1)).view(b, n, f)
        y4 = self.act_f(y4)
        y4 = self.do(y4)

        y5 = self.gc5(y4 + y1)
        b, n, f = y5.shape
        y5 = self.bn5(y5.view(b, -1)).view(b, n, f)
        y5 = self.act_f(y5)
        y5 = self.do(y5)

        return y5 + x



    def __repr__(self):
        return self.__class__.__name__ + ' (' \
               + str(self.in_features) + ' -> ' \
               + str(self.out_features) + ')'

class GC1(nn.Module):
    def __init__(self, in_features, p_dropout, bias=True, node_n=48):
        """
        Define a residual block of GCN
        """
        super(GC1, self).__init__()
        self.in_features = in_features
        self.out_features = in_features

        self.gc1 = GraphConvolution(in_features, in_features, node_n=node_n, bias=bias)
        self.bn1 = nn.BatchNorm1d(node_n * in_features)

        self.gc2 = GraphConvolution(in_features, in_features, node_n=node_n, bias=bias)
        self.bn2 = nn.BatchNorm1d(node_n * in_features)

        self.gc3 = GraphConvolution(in_features, in_features, node_n=node_n, bias=bias)
        self.bn3 = nn.BatchNorm1d(node_n * in_features)

        self.do = nn.Dropout(p_dropout)
        self.act_f = nn.Tanh()

    def forward(self, x):
        y1 = self.gc1(x)
        b, n, f = y1.shape
        y1 = self.bn1(y1.view(b, -1)).view(b, n, f)
        y1 = self.act_f(y1)
        y1 = self.do(y1)

        y2 = self.gc2(y1)
        b, n, f = y2.shape
        y2 = self.bn2(y2.view(b, -1)).view(b, n, f)
        y2 = self.act_f(y2)
        y2 = self.do(y2)

        y3 = self.gc3(y2 + y1)
        b, n, f = y3.shape
        y3 = self.bn3(y3.view(b, -1)).view(b, n, f)
        y3 = self.act_f(y3)
        y3 = self.do(y3)

        return y3 + x

class GC4stream(nn.Module):
    def __init__(self, hidden_feature, p_dropout, is_cuda, DS_stage=1, node_n=48, opt=None):
        super(GC4stream, self).__init__()
        self.is_cuda = is_cuda
        self.level_1 = [6, 7, 8, 9, 10, 11, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36,
                        37, 38, 45, 46, 47, 48, 49, 50, 51, 52, 53, 60, 61, 62, 63, 64, 65]
        self.level_1 = torch.tensor(self.level_1)
        if is_cuda:
            self.level_1 = self.level_1.cuda()
        self.level_2 = [0, 1, 2, 3, 4, 5, 12, 13, 14, 15, 16, 17, 39, 40, 41, 42, 43, 44, 54, 55, 56, 57, 58, 59]
        self.level_2 = torch.tensor(self.level_2)
        if is_cuda:
            self.level_2 = self.level_2.cuda()
        self.level_3 = []
        self.level_3 = torch.tensor(self.level_3)
        if is_cuda:
            self.level_3 = self.level_3.cuda()
        self.gcbs_1 = GC2(hidden_feature, p_dropout=p_dropout, node_n=self.level_1.shape[0])

        self.gcbs_2 = GC2(hidden_feature, p_dropout=p_dropout, node_n=self.level_2.shape[0])

        self.gcbs_3 = GC2(hidden_feature, p_dropout=p_dropout, node_n=self.level_3.shape[0])

        self.gcbs_s = GC1(hidden_feature, p_dropout=p_dropout, node_n=66)

    def forward(self, y):
        b, n, f = y.shape
        # ===========建立3个新张量===========
        y_l_1 = torch.zeros((b, self.level_1.shape[0], f))
        if self.is_cuda:
            y_l_1 = y_l_1.cuda()
        y_l_2 = torch.zeros((b, self.level_2.shape[0], f))
        if self.is_cuda:
            y_l_2 = y_l_2.cuda()
        y_l_3 = torch.zeros((b, self.level_3.shape[0], f))
        if self.is_cuda:
            y_l_3 = y_l_3.cuda()
        y_s = y
        # =======按定义3个level写入拆分的输入======
        for l_1 in range(self.level_1.shape[0]):
            y_l_1[:, l_1, :] = y[:, self.level_1[l_1], :]
        for l_2 in range(self.level_2.shape[0]):
            y_l_2[:, l_2, :] = y[:, self.level_2[l_2], :]
        for l_3 in range(self.level_3.shape[0]):
            y_l_3[:, l_3, :] = y[:, self.level_3[l_3], :]
        # ===========四路建模：3分支+1全局=========
        y_l_1 = self.gcbs_1(y_l_1)
        y_l_2 = self.gcbs_2(y_l_2)
        y_l_3 = self.gcbs_3(y_l_3)
        y_s = self.gcbs_s(y_s)
        # =====y_cat融合三分支结果再加入全局y_s=====
        y_cat = torch.zeros((b, n, f))
        if self.is_cuda:
            y_cat = y_cat.cuda()
        for l_1 in range(self.level_1.shape[0]):
            y_cat[:, self.level_1[l_1], :] = y_l_1[:, l_1, :]
        for l_2 in range(self.level_2.shape[0]):
            y_cat[:, self.level_2[l_2], :] = y_l_2[:, l_2, :]
        for l_3 in range(self.level_3.shape[0]):
            y_cat[:, self.level_3[l_3], :] = y_l_3[:, l_3, :]
        y_cat += y_s
        return y_cat


# --------------------------------------------- INCEPTION MODULE ---------------------------------------------
class IdentityAutoencoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = lambda x: x

        self.decoder = lambda x: x

    def forward(self, x):
        return x, x


class Conv1Channel(nn.Module):
    def __init__(self, nb_filters=1, filter_size=1, stride=1, dilation=1):
        super().__init__()
        self.conv = nn.Conv1d(1, nb_filters, filter_size, stride=stride, padding=0, dilation=dilation, groups=1,
                              bias=True, padding_mode='zeros')

    def forward(self, x):
        shape = x.shape
        x = x.reshape(-1, shape[-1])
        x = x[:, None, :]
        x = self.conv(x)
        x = x.reshape(shape[0], shape[1], -1)
        return x


class TemporalInceptionModule(nn.Module):
    def __init__(self):
        super().__init__()
        self.observed_length = [5, 5, 10, 10, 10]
        self.convolutions = nn.ModuleList([])

        # 5
        self.convolutions.append(Conv1Channel(nb_filters=12, filter_size=2))
        self.convolutions.append(Conv1Channel(nb_filters=9, filter_size=3))

        # 10
        self.convolutions.append(Conv1Channel(nb_filters=9, filter_size=3))
        self.convolutions.append(Conv1Channel(nb_filters=7, filter_size=5))
        self.convolutions.append(Conv1Channel(nb_filters=6, filter_size=7))

        # last frame
        self.lastconv = Conv1Channel(nb_filters=6, filter_size=3)

        self.output_size = self.forward(torch.ones(1, 1, 100)).shape[2]
        assert (len(self.observed_length) == len(self.convolutions))

    def forward(self, inpt):
        out = inpt[:, :, -10:]
        last_frame = inpt[:, :, -1]
        last_frame_seq = last_frame.repeat(10, 1, 1).permute(1, 2, 0)

        for obs_len, conv in zip(self.observed_length, self.convolutions):
            x = inpt[:, :, -obs_len:]
            y = conv(x)
            out = torch.cat((out, y), 2)
        y = self.lastconv(last_frame_seq)
        out = torch.cat((out, y), 2)

        return out


class InceptionGCN(nn.Module):
    def __init__(self, hidden_feature, p_dropout, is_cuda, DS_stage=1, node_n=48, opt=None):
        """
        :param input_feature: num of input feature
        :param hidden_feature: num of hidden feature
        :param p_dropout: drop out prob.
        :param num_stage: number of residual blocks
        :param node_n: number of nodes in graph
        """
        super().__init__()
        self.opt = opt
        self.temporal_inception_mod = TemporalInceptionModule()


        # Overwrite input parameter with correct size which depends on the TIM
        hidden_feature = self.temporal_inception_mod.output_size
        self.is_cuda = is_cuda

        self.DS_stage = DS_stage
        self.bn1 = nn.BatchNorm1d(node_n * hidden_feature)
        # self.GC4stream = GC4stream(hidden_feature=hidden_feature, p_dropout=p_dropout, is_cuda=is_cuda)

        # ====================================
        self.gcstream = []
        for i in range(self.DS_stage):
            self.gcstream.append(GC4stream(hidden_feature=hidden_feature, p_dropout=p_dropout, is_cuda=is_cuda))
        self.gcstream = nn.ModuleList(self.gcstream)

        # ====================================

        self.do = nn.Dropout(p_dropout)
        self.act_f = nn.Tanh()
        self.final = nn.Linear(hidden_feature, opt.output_n + opt.input_n)

    def forward(self, x):
        x = x[:, :, :self.opt.input_n]
        # ============阶段1=============
        y = self.temporal_inception_mod(x)
        b, n, f = y.shape
        y = self.bn1(y.view(b, -1)).view(b, n, f)
        y = self.act_f(y)
        # ============阶段2=============

        for i in range(self.DS_stage):
            y = self.gcstream[i](y)
        # ===========全连接=============
        y_cat = self.final(y)

        y_cat = y_cat + x[:, :, -1, None]

        return y_cat[:, :, self.opt.input_n:]
