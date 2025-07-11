import torch
import numpy as np
import argparse

from itertools import combinations
from utils import sgc_precompute, set_seed, load_data_pretrain
from meta import Meta
from sgc_data_generator import sgc_data_generator


def main(args):
    step = args.step
    set_seed(args.seed)

    g, idx_train, idx_valid, idx_test, class_train_dict, class_test_dict, class_valid_dict = load_data_pretrain(args.dataset)

    features = sgc_precompute(g.x, g.edge_index, args.degree)

    config = [
        ('linear', [args.hidden, features.size(1)]),
        ('linear', [args.n_way, args.hidden])
    ]

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    for i in range(args.batches):
        print("Cross Validation: {}".format((i + 1)))

        maml = Meta(args, config).to(device)

        # test_label = list(combination[i])
        # train_label = [n for n in class_label if n not in test_label]
        test_label = list(class_test_dict.keys())
        train_label = list(class_train_dict.keys())
        print('Cross Validation {} Train_Label_List: {} '.format(i + 1, train_label))
        print('Cross Validation {} Test_Label_List: {} '.format(i + 1, test_label))

        for j in range(args.epoch):
            x_spt, y_spt, x_qry, y_qry = sgc_data_generator(features, g.y, train_label, class_train_dict, args.task_num, args.n_way, args.k_spt, args.k_qry, device=device)
            accs = maml.forward(x_spt, y_spt, x_qry, y_qry)
            print('Step:', j, '\tMeta_Training_Accuracy:', accs)
            if j % 100 == 0:
                torch.save(maml.state_dict(), 'maml.pkl')
                meta_test_acc = []
                for k in range(step):
                    model_meta_trained = Meta(args, config).to(device)
                    model_meta_trained.load_state_dict(torch.load('maml.pkl'))
                    model_meta_trained.eval()
                    x_spt, y_spt, x_qry, y_qry = sgc_data_generator(features, g.y, test_label, class_test_dict, args.task_num, args.n_way, args.k_spt, args.k_qry)
                    accs = model_meta_trained.forward(x_spt, y_spt, x_qry, y_qry)
                    meta_test_acc.append(accs)
                with open(args.dataset+'.txt', 'a') as f:
                    f.write('Cross Validation:{}, Step: {}, Meta-Test_Accuracy: {}'.format(i+1, j, np.array(meta_test_acc).mean(axis=0).astype(np.float16)))
                    f.write('\n')


if __name__ == '__main__':

    argparser = argparse.ArgumentParser()

    argparser.add_argument('--epoch', type=int, help='epoch number', default=10)
    argparser.add_argument('--n_way', type=int, help='n way', default=3)
    argparser.add_argument('--meta_lr', type=float, help='meta-level outer learning rate', default=0.003)
    argparser.add_argument('--update_lr', type=float, help='task-level inner update learning rate', default=0.5)
    argparser.add_argument('--update_step', type=int, help='task-level inner update steps', default=10)
    argparser.add_argument('--update_step_test', type=int, help='update steps for finetunning', default=10)
    argparser.add_argument('--task_num', type=int, help='meta batch size, namely task num', default=5)
    argparser.add_argument('--k_spt', type=int, help='k shot for support set', default=4)
    argparser.add_argument('--k_qry', type=int, help='k shot for query set', default=12)
    argparser.add_argument('--hidden', type=int, help='Number of hidden units', default=16)
    argparser.add_argument('--batches', type=int, help='Number of batches', default=1000)

    argparser.add_argument('--dataset', type=str, default='Reddit2', help='Dataset to use.')
    argparser.add_argument('--normalization', type=str, default='AugNormAdj', help='Normalization method for the adjacency matrix.')
    argparser.add_argument('--seed', type=int, default=42, help='Random seed.')
    argparser.add_argument('--degree', type=int, default=2, help='degree of the approximation.')
    argparser.add_argument('--step', type=int, default=50, help='How many times to random select node to test')

    args = argparser.parse_args()

    main(args)
