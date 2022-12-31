import dgl
import numpy as np
import scipy.io as sio
import pickle as pkl
import json
import pathlib
import torch


dataset_source = 'ogbn-arxiv'
path_s = '../dataset/{}/'.format(dataset_source)

r''' Use this to download datasets from Pytorch Geometric library and comment out the rest of the code.


from torch_geometric.datasets import Reddit2, CoraFull
from ogb.nodeproppred.dataset_pyg import PygNodePropPredDataset

# g = Reddit2(root=path_s)[0]
# g = CoraFull(root=path_s)[0]
g = PygNodePropPredDataset(name = dataset_source, root = path_s)[0]

edge = g.edge_index
label = g.y
feat = g.x

torch.save(edge,'edge.pt')
torch.save(label,'label.pt')
torch.save(feat,'feat.pt')

'''



datas = ['Amazon_clothing', 'Amazon_electronics', 'dblp']
class_list_train,class_list_valid,class_list_test=json.load(open(path_s+'{}_class_split.json'.format(dataset_source)))

if dataset_source in datas:
    edge_index = []
    for line in open(path_s+"{}_network".format(dataset_source)):
        n1, n2 = line.strip().split('\t')
        edge_index.append([int(n1), int(n2)])
    
    edge_index = torch.tensor(edge_index).t().contiguous()
    s, t = edge_index[0], edge_index[1]

    data_train = sio.loadmat(path_s+"{}_train.mat".format(dataset_source))
    data_test = sio.loadmat(path_s+"{}_test.mat".format(dataset_source))

    num_nodes = torch.max(edge_index).data + 1
    labels = np.zeros((num_nodes,1) , dtype=int)
    labels[data_train['Index']] = data_train["Label"]
    labels[data_test['Index']] = data_test["Label"]
    labels = labels.flatten()

    features = np.zeros((num_nodes,data_train["Attributes"].shape[1]), dtype=np.float32)
    features[data_train['Index']] = data_train["Attributes"].toarray()
    features[data_test['Index']] = data_test["Attributes"].toarray()

else:
    edge_index = (torch.load('edge.pt')).numpy()
    features = (torch.load('feat.pt')).numpy()
    labels = (torch.load('label.pt')).numpy().flatten()

    s,t = edge_index[0], edge_index[1]
    num_nodes = np.amax(edge_index)+1


id_by_class = {}
class_list = np.unique(labels)
for i in class_list:
    id_by_class[i] = []
for id, cla in enumerate(labels):
    id_by_class[cla].append(id)

idx_train,idx_valid,idx_test=[],[],[]
for idx_,class_list_ in zip([idx_train,idx_valid,idx_test],[class_list_train,class_list_valid,class_list_test]):
    for class_ in class_list_:
        idx_.extend(id_by_class[class_])

path_target = './G-Meta_Data/'+dataset_source+'/'
pathlib.Path(path_target).mkdir(parents=True, exist_ok=True)

with open(path_target+'train.csv','w') as f:
    f.write(',name,label\n')
    for i in range(len(idx_train)):
        f.write(str(i)+',0_'+str(idx_train[i])+','+str(labels[idx_train[i]])+'\n')

with open(path_target+'val.csv','w') as f:
    f.write(',name,label\n')
    for i in range(len(idx_valid)):
        f.write(str(i)+',0_'+str(idx_valid[i])+','+str(labels[idx_valid[i]])+'\n')
        
with open(path_target+'test.csv','w') as f:
    f.write(',name,label\n')
    for i in range(len(idx_test)):
        f.write(str(i)+',0_'+str(idx_test[i])+','+str(labels[idx_test[i]])+'\n')

with open(path_target+'features.npy', 'wb') as f:
    np.save(f, features)

g = [dgl.DGLGraph((s,t))]
with open(path_target+'graph_dgl.pkl', 'wb') as f:
    pkl.dump(g,f)

labels = {'0_'+str(k):v for k,v in zip(range(num_nodes) , labels)}
with open(path_target+'label.pkl', 'wb') as f:
    pkl.dump(labels,f)