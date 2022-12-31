import random
import torch

def sgc_data_generator(features, labels, s_labels ,class_train_dict, task_num, n_way, k_spt, k_qry, device='cuda'):
    x_spt = []
    y_spt = []
    x_qry = []
    y_qry = []

    labels_local = labels.clone().detach()
    select_class = random.sample(s_labels, n_way)
    d = {k:v for k,v in zip(select_class,range(n_way))}
    labels_local = torch.tensor([torch.tensor(d[labels_local[i].item()]) if labels_local[i] in select_class else labels_local[i] for i in range(len(labels_local))])
    for _ in range(task_num):
        spt_idx, qry_idx = [],[]
        for i in select_class:
            idx_ = random.sample(class_train_dict[i], k_spt+k_qry)
            spt_idx.extend(idx_[:k_spt])
            qry_idx.extend(idx_[k_spt:k_qry])

        random.shuffle(spt_idx)
        random.shuffle(qry_idx)
          
        x_spt.append(features[spt_idx].to(device))
        y_spt.append(labels_local[spt_idx].to(device))
        x_qry.append(features[qry_idx].to(device))
        y_qry.append(labels_local[qry_idx].to(device))

    return x_spt, y_spt, x_qry, y_qry