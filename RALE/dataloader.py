import numpy as np
import scipy.io as sio
import pathlib
import json
import torch


dataset_source = 'corafull'
path_s = '../dataset/{}/'.format(dataset_source)


r''' Use this to download datasets from Pytorch Geometric library and comment out the rest of the code.


from torch_geometric.datasets import Reddit2, CoraFull
from ogb.nodeproppred.dataset_pyg import PygNodePropPredDataset

# g = Reddit2(root=path_s)[0]
g = CoraFull(root=path_s)[0]
# g = PygNodePropPredDataset(name = dataset_source, root = path_s)[0]

edge = g.edge_index
label = g.y
feat = g.x

torch.save(edge,'edge.pt')
torch.save(label,'label.pt')
torch.save(feat,'feat.pt')

'''


datas = ['dblp','Amazon_clothing','Amazon_electronics']
class_list_train,class_list_valid,class_list_test=json.load(open(path_s+'{}_class_split.json'.format(dataset_source)))

if dataset_source in datas:
    path_s = '../dataset/{}/'.format(dataset_source)
    edge_index = []
    for line in open(path_s+"{}_network".format(dataset_source)):
        n1, n2 = line.strip().split('\t')
        edge_index.append([int(n1), int(n2)])

    data_train = sio.loadmat(path_s+"{}_train.mat".format(dataset_source))
    data_test = sio.loadmat(path_s+"{}_test.mat".format(dataset_source))

    num_nodes = np.amax(edge_index) + 1
    labels = np.zeros((num_nodes,1), dtype=int)
    labels[data_train['Index']] = data_train["Label"]
    labels[data_test['Index']] = data_test["Label"]
    labels = labels.flatten()

    features = np.zeros((num_nodes,data_train["Attributes"].shape[1]))
    features[data_train['Index']] = data_train["Attributes"].toarray()
    features[data_test['Index']] = data_test["Attributes"].toarray()

else:
    edge_index = (torch.load('edge.pt')).t().numpy()
    features = (torch.load('feat.pt')).numpy()
    labels = (torch.load('label.pt')).numpy().flatten()
    num_nodes = np.amax(edge_index)+1


path = './datasets/'+dataset_source
pathlib.Path(path).mkdir(parents=True, exist_ok=True)

with open(path+'/graph.node', 'w') as f:
        f.write(str(num_nodes) + '\t' + str(features.shape[1]) + '\n')
        for i in range(len(features)):
            feat_str = features[i].tolist()
            feat_str = '\t'.join(list(map(str, feat_str)))
            s = str(i) + '\t' + str(labels[i]) + '\t' + feat_str + '\n'
            f.write(s)

with open(path+'/graph.edge', 'w') as f:
        for i in range(len(edge_index)):
            u,v = str(edge_index[i][0]), str(edge_index[i][1])
            f.write(u+'\t'+v+'\n')

train_no, val_no, test_no = len(class_list_train), len(class_list_valid), len(class_list_test)

paths = path+'/datasets-splits/'
pathlib.Path(paths).mkdir(parents=True, exist_ok=True)

for i in range(10):
    perm = np.random.permutation(np.unique(labels))
    train,val,test = perm[0:train_no],perm[train_no:train_no+val_no],perm[train_no+val_no:train_no+val_no+test_no]

    with open(paths+'train-class-'+str(i), 'w') as f:
        v = list(map(str, train.tolist()))
        f.write('\n'.join(v))

    with open(paths+'val-class-'+str(i), 'w') as f:
        v = list(map(str, val.tolist()))
        f.write('\n'.join(v))

    with open(paths+'test-class-'+str(i), 'w') as f:
        v = list(map(str, test.tolist()))
        f.write('\n'.join(v))