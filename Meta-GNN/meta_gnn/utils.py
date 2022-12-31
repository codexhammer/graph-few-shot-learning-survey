import numpy as np
import scipy.sparse as sp
import torch
import sys
import pickle as pkl
import networkx as nx
from normalization import fetch_normalization, row_normalize
from sklearn.metrics import f1_score
import json
from collections import defaultdict
import scipy.io as sio
from torch_geometric.data import Data


def f1(output, labels):
    output = output.max(1)[1]
    output = output.cpu().detach().numpy()
    labels = labels.cpu().detach().numpy()
    micro = f1_score(labels, output, average='micro')
    return micro


def parse_index_file(filename):
    index = []
    for line in open(filename):
        index.append(int(line.strip()))
    return index


def preprocess_citation(adj, features, normalization="AugNormAdj"):
    adj_normalizer = fetch_normalization(normalization)
    adj = adj_normalizer(adj)
    features = row_normalize(features)
    return adj, features


def sparse_mx_to_torch_sparse_tensor(sparse_mx):
    sparse_mx = sparse_mx.tocoo().astype(np.float32)
    indices = torch.from_numpy(
        np.vstack((sparse_mx.row, sparse_mx.col)).astype(np.int64))
    values = torch.from_numpy(sparse_mx.data)
    shape = torch.Size(sparse_mx.shape)
    return torch.sparse.FloatTensor(indices, values, shape)

def to_torch_coo_tensor(edge_index, edge_attr = None,size = None):
    if size is None:
        size = int(edge_index.max()) + 1
    if not isinstance(size, (tuple, list)):
        size = (size, size)

    if edge_attr is None:
        edge_attr = torch.ones(edge_index.size(1), device=edge_index.device)

    size = tuple(size) + edge_attr.size()[1:]
    out = torch.sparse_coo_tensor(edge_index, edge_attr, size,
                                  device=edge_index.device)
    out = out.coalesce()
    return out

def load_data_pretrain(dataset_source):
    path = '../../dataset/{}/'.format(dataset_source)
    preload_data = ['Amazon_clothing', 'Amazon_electronics', 'dblp']
    class_list_train,class_list_valid,class_list_test=json.load(open(path+'{}_class_split.json'.format(dataset_source)))

    if dataset_source in preload_data:
        edge_index = []
        for line in open(path+"{}_network".format(dataset_source)):
            n1, n2 = line.strip().split('\t')
            edge_index.append([int(n1), int(n2)])

        data_train = sio.loadmat(path+"{}_train.mat".format(dataset_source))
        data_test = sio.loadmat(path+"{}_test.mat".format(dataset_source))

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
        edge_index = torch.tensor(edge_index).t().contiguous()
        edge_index = to_torch_coo_tensor(edge_index)
        g = Data(edge_index=edge_index , x=features, y=labels, train_mask=train_mask, val_mask=val_mask, test_mask=test_mask )
        print('nodes num',num_nodes)

    elif dataset_source=='cora-full':
        from torch_geometric.datasets import CoraFull
        g = CoraFull(root=path+'/Corafull')[0]
        g.train_mask = torch.isin(g.y , torch.tensor(class_list_train))
        g.val_mask = torch.isin(g.y , torch.tensor(class_list_valid))
        g.test_mask = torch.isin(g.y , torch.tensor(class_list_test))
        g.edge_index = to_torch_coo_tensor(g.edge_index)
        num_nodes = g.num_nodes
        print('nodes num',num_nodes)

    elif dataset_source=='ogbn-arxiv':

        from ogb.nodeproppred.dataset_pyg import PygNodePropPredDataset

        g = PygNodePropPredDataset(name = dataset_source, root = path+'/ogbn/')[0]
        g.train_mask = torch.isin(g.y , torch.tensor(class_list_train))
        g.val_mask = torch.isin(g.y , torch.tensor(class_list_valid))
        g.test_mask = torch.isin(g.y , torch.tensor(class_list_test))
        g.edge_index = to_torch_coo_tensor(g.edge_index)
        num_nodes = g.num_nodes
        print('nodes num',num_nodes)
        
    elif dataset_source=='Reddit2':
        from torch_geometric.datasets import Reddit2
        g = Reddit2(root=path+'/Reddit2/')[0]
        g.train_mask = torch.isin(g.y , torch.tensor(class_list_train))
        g.val_mask = torch.isin(g.y , torch.tensor(class_list_valid))
        g.test_mask = torch.isin(g.y , torch.tensor(class_list_test))
        g.edge_index = to_torch_coo_tensor(g.edge_index)
        num_nodes = g.num_nodes
        print('nodes num',num_nodes)

    g.x = torch.nn.functional.normalize(g.x, p=2, dim=1) 
    id_by_class = {}
    class_list = torch.unique(g.y).tolist()
    for i in class_list:
        id_by_class[i] = []
    for id, cla in enumerate(g.y.tolist()):
        id_by_class[cla].append(id)

    idx_train,idx_valid,idx_test=[],[],[]
    for idx_,class_list_ in zip([idx_train,idx_valid,idx_test],[class_list_train,class_list_valid,class_list_test]):
        for class_ in class_list_:
            idx_.extend(id_by_class[class_])

    class_train_dict=defaultdict(list)
    for one in class_list_train:
        for i,label in enumerate(g.y.tolist()):
            if label==one:
                class_train_dict[one].append(i)

    class_valid_dict = defaultdict(list)
    for one in class_list_valid:
        for i, label in enumerate(g.y.tolist()):
            if label == one:
                class_valid_dict[one].append(i)

    class_test_dict = defaultdict(list)
    for one in class_list_test:
        for i, label in enumerate(g.y.tolist()):
            if label == one:
                class_test_dict[one].append(i)

    return g, idx_train, idx_valid, idx_test, class_train_dict, class_test_dict, class_valid_dict

def sgc_precompute(features, adj, degree):
    for i in range(degree):
        features = torch.spmm(adj, features)
    return features


def set_seed(seed, cuda=True):
    np.random.seed(seed)
    torch.manual_seed(seed)
    if cuda:
        torch.cuda.manual_seed(seed)

