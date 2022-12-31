import numpy as np
import torch
import scipy.io as sio
import random
from sklearn.metrics import f1_score
from torch_geometric.data import Data
from torch_geometric.utils import degree
import json


def load_data(dataset_source):
    class_list_train,class_list_valid,class_list_test=json.load(open('../dataset/'+dataset_source+'/{}_class_split.json'.format(dataset_source)))
    pre_load_data = ['Amazon_clothing', 'Amazon_electronics', 'dblp']

    if dataset_source in pre_load_data:
        edge_index = []
        for line in open("../dataset/"+dataset_source+"/{}_network".format(dataset_source)):
            n1, n2 = line.strip().split('\t')
            edge_index.append([int(n1), int(n2)])

        data_train = sio.loadmat("../dataset/"+dataset_source+"/{}_train.mat".format(dataset_source))
        data_test = sio.loadmat("../dataset/"+dataset_source+"/{}_test.mat".format(dataset_source))

        num_nodes = np.amax(edge_index) + 1
        labels = np.zeros((num_nodes,1))
        labels[data_train['Index']] = data_train["Label"]
        labels[data_test['Index']] = data_test["Label"]
        labels = torch.LongTensor(labels).squeeze()
        
        train_mask = torch.isin(labels, torch.tensor(class_list_train))
        val_mask = torch.isin(labels, torch.tensor(class_list_valid))
        test_mask = torch.isin(labels, torch.tensor(class_list_test))

        features = np.zeros((num_nodes,data_train["Attributes"].shape[1]))
        features[data_train['Index']] = data_train["Attributes"].toarray()
        features[data_test['Index']] = data_test["Attributes"].toarray()
        features = torch.FloatTensor(features)

        print('nodes num',num_nodes)
        edge_index = torch.tensor(edge_index).t().contiguous()
        g = Data(x=features, edge_index=edge_index, y=labels, 
                train_mask=train_mask, val_mask=val_mask, test_mask=test_mask)
        
    elif dataset_source == 'cora_full':
        from torch_geometric.datasets import CoraFull
        g = CoraFull(root='../dataset/Corafull')[0]
        g.train_mask = torch.isin(g.y , torch.tensor(class_list_train))
        g.val_mask = torch.isin(g.y , torch.tensor(class_list_valid))
        g.test_mask = torch.isin(g.y , torch.tensor(class_list_test))

    elif dataset_source == 'ogbn-arxiv':
        from torch_geometric.datasets import Reddit2
        g = Reddit2(root='./dataset/Reddit/')[0]
        g.train_mask = torch.isin(g.y , torch.tensor(class_list_train))
        g.val_mask = torch.isin(g.y , torch.tensor(class_list_valid))
        g.test_mask = torch.isin(g.y , torch.tensor(class_list_test))

    g.degree_in = degree(g.edge_index[0], num_nodes)

    id_by_class = {}
    class_list = torch.unique(g.y).tolist()
    for i in class_list:
        id_by_class[i] = []
    for id, cla in enumerate(g.y.tolist()):
        id_by_class[cla].append(id)

    return g, class_list_train, class_list_valid, class_list_test, id_by_class 


# def normalize(mx):
#     """Row-normalize sparse matrix"""
#     rowsum = np.array(mx.sum(1))
#     r_inv = np.power(rowsum, -1).flatten()
#     r_inv[np.isinf(r_inv)] = 0.
#     r_mat_inv = sp.diags(r_inv)
#     mx = r_mat_inv.dot(mx)
#     return mx


# def normalize_adj(adj):
#     """Symmetrically normalize adjacency matrix."""
#     adj = sp.coo_matrix(adj)
#     rowsum = np.array(adj.sum(1))
#     d_inv_sqrt = np.power(rowsum, -0.5).flatten()
#     d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.
#     d_mat_inv_sqrt = sp.diags(d_inv_sqrt)
#     return adj.dot(d_mat_inv_sqrt).transpose().dot(d_mat_inv_sqrt).tocoo()


def accuracy(output, labels):
    preds = output.max(1)[1].type_as(labels)
    correct = preds.eq(labels).double()
    correct = correct.sum()
    return correct / len(labels)


def f1(output, labels):
    preds = output.max(1)[1].type_as(labels)
    f1 = f1_score(labels, preds, average='weighted')
    return f1


# def sparse_mx_to_torch_sparse_tensor(sparse_mx):
#     """Convert a scipy sparse matrix to a torch sparse tensor."""
#     sparse_mx = sparse_mx.tocoo().astype(np.float32)
#     indices = torch.from_numpy(
#         np.vstack((sparse_mx.row, sparse_mx.col)).astype(np.int64))
#     values = torch.from_numpy(sparse_mx.data)
#     shape = torch.Size(sparse_mx.shape)
#     return torch.sparse.FloatTensor(indices, values, shape)



def task_generator(id_by_class, class_list, n_way, k_shot, m_query):

    # sample class indices
    class_selected = random.sample(class_list, n_way)
    id_support = []
    id_query = []
    for cla in class_selected:
        temp = random.sample(id_by_class[cla], k_shot + m_query)
        id_support.extend(temp[:k_shot])
        id_query.extend(temp[k_shot:])

    return np.array(id_support), np.array(id_query), class_selected



def euclidean_dist(x, y):
    # x: N x D query
    # y: M x D prototype
    n = x.size(0)
    m = y.size(0)
    d = x.size(1)
    assert d == y.size(1)

    x = x.unsqueeze(1).expand(n, m, d)
    y = y.unsqueeze(0).expand(n, m, d)

    return torch.pow(x - y, 2).sum(2)  # N x M