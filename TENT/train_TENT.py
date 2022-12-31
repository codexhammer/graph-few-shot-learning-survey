import numpy as np
import random
import torch
import numpy as np
import json
from collections import defaultdict
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F
import argparse
import time
import scipy.io as sio
import random
from sklearn.metrics import f1_score
import json
from sklearn.linear_model import LogisticRegression
from sklearn import metrics
from base_model_SSL import GCN_dense
from base_model_SSL import Linear
from base_model_SSL import GCN_emb
from torch_geometric.data import Data


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


def cal_euclidean(input):
    return torch.cdist(input.unsqueeze(0),input.unsqueeze(0)).squeeze()

def load_data_pretrain(dataset_source):
    path_s = '../dataset/{}/'.format(dataset_source)
    preload_data = ['Amazon_clothing', 'Amazon_electronics', 'dblp']
    class_list_train,class_list_valid,class_list_test=json.load(open(path_s+'{}_class_split.json'.format(dataset_source)))

    if dataset_source in preload_data:
        edge_index = []
        for line in open(path_s+"{}_network".format(dataset_source)):
            n1, n2 = line.strip().split('\t')
            edge_index.append([int(n1), int(n2)])

        data_train = sio.loadmat(path_s+"{}_train.mat".format(dataset_source))
        data_test = sio.loadmat(path_s+"{}_test.mat".format(dataset_source))

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

    elif dataset_source=='corafull':
        from torch_geometric.datasets import CoraFull
        g = CoraFull(root=path_s)[0]
        g.train_mask = torch.isin(g.y , torch.tensor(class_list_train))
        g.val_mask = torch.isin(g.y , torch.tensor(class_list_valid))
        g.test_mask = torch.isin(g.y , torch.tensor(class_list_test))
                        
        # class_list =  class_list_train+class_list_valid+class_list_test

        # id_by_class = {}
        # for i in class_list:
        #     id_by_class[i] = []
        # for id, cla in enumerate(g.y.tolist()):
        #     id_by_class[cla].append(id)
        
    elif dataset_source=='ogbn-arxiv':

        from ogb.nodeproppred.dataset_pyg import PygNodePropPredDataset

        dataset = PygNodePropPredDataset(name = dataset_source, root =path_s)

        split_idx = dataset.get_idx_split()
        train_mask, valid_mask, test_mask = split_idx["train"], split_idx["valid"], split_idx["test"]
        g = dataset[0]
        g.train_mask, g.val_mask, g.test_mask = train_mask, valid_mask, test_mask

        # n1s=graph['edge_index'][0]
        # n2s=graph['edge_index'][1]

        num_nodes = g.num_nodes
        print('nodes num',num_nodes)
        # adj = sp.coo_matrix((np.ones(len(n1s)), (n1s, n2s)),
        #                         shape=(num_nodes, num_nodes))    
        # degree = np.sum(adj, axis=1)
        # degree = torch.FloatTensor(degree)
        # adj = normalize(adj + sp.eye(adj.shape[0]))
        # adj = sparse_mx_to_torch_sparse_tensor(adj)

        # features=torch.FloatTensor(graph['node_feat'])
        # labels=torch.LongTensor(labels).squeeze()
    
    elif dataset_source=='Reddit2':
        from torch_geometric.datasets import Reddit2
        g = Reddit2(root=path_s)[0]

        
    #     class_list =  class_list_train+class_list_valid+class_list_test

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


def InforNCE_Loss(anchor, sample, tau, all_negative=False, temperature_matrix=None):
    def _similarity(h1: torch.Tensor, h2: torch.Tensor):
        h1 = F.normalize(h1)
        h2 = F.normalize(h2)
        return h1 @ h2.t()

    assert anchor.shape[0] == sample.shape[0]

    
    pos_mask = torch.eye(anchor.shape[0], dtype=torch.float)
    if dataset!='ogbn-arxiv':
        pos_mask=pos_mask.cuda()
    
    neg_mask = 1. - pos_mask

    sim = _similarity(anchor, sample / temperature_matrix if temperature_matrix != None else sample) / tau
    exp_sim = torch.exp(sim) * (pos_mask + neg_mask)

    if not all_negative:
        log_prob = sim - torch.log(exp_sim.sum(dim=1, keepdim=True))
    else:
        log_prob = - torch.log(exp_sim.sum(dim=1, keepdim=True))

    loss = log_prob * pos_mask
    loss = loss.sum(dim=1) / pos_mask.sum(dim=1)

    return -loss.mean(), sim




parser = argparse.ArgumentParser()
parser.add_argument('--use_cuda', action='store_true', default=True, help='Disables CUDA training.')
parser.add_argument('--seed', type=int, default=1234, help='Random seed.')
parser.add_argument('--epochs', type=int, default=2000,
                    help='Number of epochs to train.')
parser.add_argument('--test_epochs', type=int, default=100,
                    help='Number of epochs to train.')
parser.add_argument('--lr', type=float, default=0.05,
                    help='Initial learning rate.')
parser.add_argument('--batch_size', type=int, default=64,
                    help='Number of hidden units.')
parser.add_argument('--weight_decay', type=float, default=5e-4,  # 5e-4
                    help='Weight decay (L2 loss on parameters).')
parser.add_argument('--hidden1', type=int, default=16,
                    help='Number of hidden units.')
parser.add_argument('--hidden2', type=int, default=16,
                    help='Number of hidden units.')
parser.add_argument('--dropout', type=float, default=0.2,
                    help='Dropout rate (1 - keep probability).')


args = parser.parse_args(args=[])


random.seed(args.seed)
np.random.seed(args.seed)
torch.manual_seed(args.seed)
if args.use_cuda:
    torch.cuda.manual_seed(args.seed)

loss_f = nn.CrossEntropyLoss()


Q=10


fine_tune_steps = 20
fine_tune_lr = 0.1


results=defaultdict(dict)


for dataset in ['Amazon_electronics', 'corafull', 'ogbn-arxiv','dblp']:

    g, idx_train, idx_valid, idx_test, class_train_dict, class_test_dict, class_valid_dict = load_data_pretrain(dataset)
    # g = load_data_pretrain(dataset)

    # adj = adj_sparse.to_dense()
    # if dataset!='ogbn-arxiv':
    g = g.cuda()
    # else:
    #     args.use_cuda=False

    N_set=[5,10]
    K_set=[3,5]

    for N in N_set:
        for K in K_set:
            for repeat in range(5):
                print('done')
                print(dataset)
                print('N={},K={}'.format(N,K))

                model = GCN_dense(nfeat=args.hidden1,
                                  nhid=args.hidden2,
                                  nclass=g.y.max().item() + 1,
                                  dropout=args.dropout)

                GCN_model=GCN_emb(nfeat=g.x.shape[1],
                            nhid=args.hidden1,
                            nclass=g.y.max().item() + 1,
                            dropout=args.dropout)

                classifier = Linear(args.hidden1, g.y.max().item() + 1)

                optimizer = optim.Adam([{'params': model.parameters()}, {'params': classifier.parameters()},{'params': GCN_model.parameters()}],
                                       lr=args.lr, weight_decay=args.weight_decay)


                support_labels=torch.zeros(N*K,dtype=torch.long)
                for i in range(N):
                    support_labels[i * K:(i + 1) * K] = i
                query_labels=torch.zeros(N*Q,dtype=torch.long)
                for i in range(N):
                    query_labels[i * Q:(i + 1) * Q] = i

                if args.use_cuda:
                    model.cuda()
                    # features = features.cuda()
                    GCN_model.cuda()
                    # adj_sparse = adj_sparse.cuda()
                    g.y = g.y.cuda()
                    classifier.cuda()

                    support_labels=support_labels.cuda()
                    query_labels=query_labels.cuda()

                def pre_train(epoch, N, mode='train'):

                    if mode == 'train':
                        model.train()
                        optimizer.zero_grad()
                    else:
                        model.eval()
                    # emb_features=GCN_model(features, adj_sparse)
                    emb_features=GCN_model(g.x, g.edge_index)

                    target_idx = []
                    target_graph_adj_and_feat = []

                    pos_node_idx = []

                    if mode == 'train':
                        class_dict = class_train_dict
                    elif mode == 'test':
                        class_dict = class_test_dict
                    elif mode=='valid':
                        class_dict = class_valid_dict
                    

                    classes = np.random.choice(list(class_dict.keys()), N, replace=False).tolist()

                    pos_graph_adj_and_feat=[]   
                    for i in classes:
                        # sample from one specific class
                        sampled_idx=np.random.choice(class_dict[i], K+Q, replace=False).tolist()
                        pos_node_idx.extend(sampled_idx[:K])
                        target_idx.extend(sampled_idx[K:])

                        class_pos_idx=sampled_idx[:K]

                        if K==1 and torch.nonzero(adj[class_pos_idx,:]).shape[0]==1:
                            pos_class_graph_adj=adj[class_pos_idx,class_pos_idx].reshape([1,1])
                            pos_graph_feat=emb_features[class_pos_idx]
                        else:
                            pos_graph_neighbors = torch.nonzero(adj[class_pos_idx, :].sum(0)).squeeze()


                            pos_graph_adj = adj[pos_graph_neighbors, :][:, pos_graph_neighbors]


                            pos_class_graph_adj=torch.eye(pos_graph_neighbors.shape[0]+1,dtype=torch.float)

                            pos_class_graph_adj[1:,1:]=pos_graph_adj

                            pos_graph_feat=torch.cat([emb_features[class_pos_idx].mean(0,keepdim=True),emb_features[pos_graph_neighbors]],0)


                        if dataset!='ogbn-arxiv':
                            pos_class_graph_adj=pos_class_graph_adj.cuda()

                        pos_graph_adj_and_feat.append((pos_class_graph_adj, pos_graph_feat))


                    target_graph_adj_and_feat=[]  
                    for node in target_idx:
                        if torch.nonzero(adj[node,:]).shape[0]==1:
                            pos_graph_adj=adj[node,node].reshape([1,1])
                            pos_graph_feat=emb_features[node].unsqueeze(0)
                        else:
                            pos_graph_neighbors = torch.nonzero(adj[node, :]).squeeze()
                            pos_graph_neighbors = torch.nonzero(adj[pos_graph_neighbors, :].sum(0)).squeeze()
                            pos_graph_adj = adj[pos_graph_neighbors, :][:, pos_graph_neighbors]
                            pos_graph_feat = emb_features[pos_graph_neighbors]



                        target_graph_adj_and_feat.append((pos_graph_adj, pos_graph_feat))



                    class_generate_emb=torch.stack([sub[1][0] for sub in pos_graph_adj_and_feat],0).mean(0)


                    parameters=model.generater(class_generate_emb)


                    gc1_parameters=parameters[:(args.hidden1+1)*args.hidden2*2]
                    gc2_parameters=parameters[(args.hidden1+1)*args.hidden2*2:]

                    gc1_w=gc1_parameters[:args.hidden1*args.hidden2*2].reshape([2,args.hidden1,args.hidden2])
                    gc1_b=gc1_parameters[args.hidden1*args.hidden2*2:].reshape([2,args.hidden2])

                    gc2_w=gc2_parameters[:args.hidden2*args.hidden2*2].reshape([2,args.hidden2,args.hidden2])
                    gc2_b=gc2_parameters[args.hidden2*args.hidden2*2:].reshape([2,args.hidden2])


                    model.eval()
                    ori_emb = []
                    for i, one in enumerate(target_graph_adj_and_feat):
                        sub_adj, sub_feat = one[0], one[1]
                        ori_emb.append(model(sub_feat, sub_adj, gc1_w, gc1_b, gc2_w, gc2_b).mean(0))  # .mean(0))

                    target_embs = torch.stack(ori_emb, 0)

                    class_ego_embs=[]
                    for sub_adj, sub_feat in pos_graph_adj_and_feat:
                        class_ego_embs.append(model(sub_feat,sub_adj,gc1_w,gc1_b,gc2_w,gc2_b)[0])
                    class_ego_embs=torch.stack(class_ego_embs,0)



                    target_embs=target_embs.reshape([N,Q,-1]).transpose(0,1)
                    
                    support_features = emb_features[pos_node_idx].reshape([N,K,-1])
                    class_features=support_features.mean(1)
                    taus=[]
                    for j in range(N):
                        taus.append(torch.linalg.norm(support_features[j]-class_features[j],-1).sum(0))
                    taus=torch.stack(taus,0)

                        
                    similarities=[]
                    for j in range(Q):
                        class_contras_loss, similarity=InforNCE_Loss(target_embs[j],class_ego_embs/taus.unsqueeze(-1),tau=0.5)
                        similarities.append(similarity)

                    loss_supervised=loss_f(classifier(emb_features[idx_train]), g.y[idx_train])

                    loss=loss_supervised


                    labels_train=g.y[target_idx]
                    for j, class_idx in enumerate(classes[:N]):
                        labels_train[labels_train==class_idx]=j
                        
                    loss+=loss_f(torch.stack(similarities,0).transpose(0,1).reshape([N*Q,-1]), labels_train)
                    
                    
                    acc_train = accuracy(torch.stack(similarities,0).transpose(0,1).reshape([N*Q,-1]), labels_train)

                    if mode=='valid' or mode=='test' or (mode=='train' and epoch%250==249):
                        support_features = torch.norm(emb_features[pos_node_idx] , dim=1).detach().cpu().numpy()
                        query_features = torch.norm(emb_features[target_idx], dim=1).detach().cpu().numpy()

                        support_labels=torch.zeros(N*K,dtype=torch.long)
                        for i in range(N):
                            support_labels[i * K:(i + 1) * K] = i

                        query_labels=torch.zeros(N*Q,dtype=torch.long)
                        for i in range(N):
                            query_labels[i * Q:(i + 1) * Q] = i


                        clf = LogisticRegression(penalty='l2',
                                                 random_state=0,
                                                 C=1.0,
                                                 solver='lbfgs',
                                                 max_iter=1000,
                                                 multi_class='multinomial')
                        clf.fit(support_features, support_labels.numpy())
                        query_ys_pred = clf.predict(query_features)

                        acc_train = metrics.accuracy_score(query_labels, query_ys_pred)





                    if mode == 'train':
                        loss.backward()
                        optimizer.step()

                    if epoch % 250 == 249 and mode == 'train':
                        print('Epoch: {:04d}'.format(epoch + 1),
                              'loss_train: {:.4f}'.format(loss.item()),
                              'acc_train: {:.4f}'.format(acc_train.item()))
                    return acc_train.item()

                # Train model
                t_total = time.time()
                best_acc = 0
                best_valid_acc=0
                count=0
                for epoch in range(args.epochs):
                    acc_train=pre_train(epoch, N=N)
                    

                    if  epoch > 0 and epoch % 50 == 0:

                        temp_accs=[]
                        for epoch_test in range(50):
                            temp_accs.append(pre_train(epoch_test, N=N, mode='test'))

                        accs = []

                        for epoch_test in range(50):
                            accs.append(pre_train(epoch_test, N=N if dataset!='ogbn-arxiv' else 5, mode='valid'))

                        valid_acc=np.array(accs).mean(axis=0)
                        print("Epoch: {:04d} Meta-valid_Accuracy: {:.4f}".format(epoch + 1, valid_acc))


                        if valid_acc>best_valid_acc:
                            best_test_accs=temp_accs
                            best_valid_acc=valid_acc
                            count=0
                        else:
                            count+=1
                            if count>=10:
                                break


                accs=best_test_accs

                print('Test Acc',np.array(accs).mean(axis=0))
                results[dataset]['{}-way {}-shot {}-repeat'.format(N,K,repeat)]=[np.array(accs).mean(axis=0)]

                json.dump(results[dataset],open('./TENT-result_{}.json'.format(dataset),'w'))


            accs=[]
            for repeat in range(5):
                accs.append(results[dataset]['{}-way {}-shot {}-repeat'.format(N,K,repeat)][0])


            results[dataset]['{}-way {}-shot'.format(N,K)]=[np.mean(accs)]
            results[dataset]['{}-way {}-shot_print'.format(N,K)]='acc: {:.4f}'.format(np.mean(accs))


            json.dump(results[dataset],open('./TENT-result_{}.json'.format(dataset),'w'))   

            del model

    del g