#!/usr/bin/env python
# -*- coding: UTF-8 -*-

import os
import time
import argparse
import numpy as np
import torch
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
from utils import SoftCrossEntropy, extract_dataset, seed
from utils import get_lambda, mixup_data, mixup_criterion, estimate_kde
from sklearn.datasets import make_moons, make_circles
from sklearn.metrics import precision_score
from sklearn import svm

parser = argparse.ArgumentParser(description='synthetic dataset')
parser.add_argument('-n', '--network', default='fc', type=str,
                    help='network used in training')
parser.add_argument('-t', '--train_setting', default=1, type=int,
                    help='mode for training setting, 0: testing, 1: training')
parser.add_argument('-m', '--mixup_setting', default=4, type=int,
                    help='mode for mixup setting, 0: vanilla training, 1: mixup, 2: mixup+genlabel')
parser.add_argument('-b', '--batch_size', default=128, type=int,
                    help='batch_size')
parser.add_argument('-l', '--lr', default=0.1, type=float,
                    help='learning rate')
parser.add_argument('-e', '--epoch', default=100, type=int,
                    help='number of training epochs')
parser.add_argument('--trial', default=1, type=int,
                    help='trial number (1,2, ...), will change random seed')
parser.add_argument('--dir_num', default=1, type=int,
                    help='directory number, suffix for log & model')
parser.add_argument('--dataset', default='twocircle', type=str,
                    help='name of dataset to use, |moon|circle|twocircle|')
parser.add_argument('--lam', default=1, type=float,
                    help='loss ratio')
parser.add_argument('-v', '--validate', default='cln', type=str,
                    help='how to validate, |cln|')
parser.add_argument('--bw', default=0.2, type=float,
                    help='band width for KDE')
parser.add_argument('--num_sample', default=1000, type=int,
                    help='number of samples for training and testing')

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def get_data(dataset, digit_list, train=True):

    onehot = torch.eye(2)

    if train:
        num_sample = 1000
    else:
        num_sample = 1000

    args.num_sample = num_sample

    if dataset == 'moon':
        train_data, train_label = make_moons(n_samples=num_sample)
        noise = 0.1
    elif dataset == 'circle':
        train_data, train_label = make_circles(n_samples=num_sample)
        noise = 0.02
    elif dataset == 'twocircle':
        train_data_1, train_label_1 = make_circles(n_samples=num_sample)
        train_data_2, train_label_2 = make_circles(n_samples=num_sample)

        train_data_1 = train_data_1 - np.asarray([-1.2, 0]).reshape(1, 2)
        train_data_2 = train_data_2 + np.asarray([-1.2, 0]).reshape(1, 2)

        train_label_2 = np.abs(train_label_2 - 1)

        train_data = np.concatenate((train_data_1, train_data_2), axis=0)
        train_label = np.concatenate((train_label_1, train_label_2), axis=0)
        noise = 0.01
    else:
        raise NotImplementedError

    train_data += np.random.laplace(scale=noise, size=train_data.shape)

    # convert to tensor
    train_data, train_label = torch.from_numpy(train_data).float(), torch.from_numpy(train_label).long()

    plt.scatter(train_data[:, 0], train_data[:, 1], c=train_label, alpha=0.5)

    plt.axis('off')
    plt.axis('equal')

    if train:
        plt.savefig('Syn_results/train_data_{}_mode_{}_sample_{}.png'.format(dataset, args.mixup_setting, args.num_sample))
    else:
        plt.savefig('Syn_results/test_data_{}_mode_{}_sample_{}.png'.format(dataset, args.mixup_setting, args.num_sample))
    plt.close()

    train_label = onehot[train_label]

    dataset = torch.utils.data.TensorDataset(train_data, train_label)

    return dataset


def decision_boundary(net, test_dataset):

    net.eval()

    prediction_list = []
    data_list = []

    if args.dataset == 'moon':
        scale = torch.Tensor([3, 3]).view(1, 2)
        bias = torch.Tensor([-1, -1.25]).view(1, 2)
    elif args.dataset == 'circle':
        scale = torch.Tensor([3, 3]).view(1, 2)
        bias = torch.Tensor([-1.5, -1.5]).view(1, 2)
    elif args.dataset == 'twocircle':
        scale = torch.Tensor([5, 3]).view(1, 2)
        bias = torch.Tensor([-2.5, -1.5]).view(1, 2)
    data = torch.rand((100000, 2)) * scale +bias

    dataset = torch.utils.data.TensorDataset((data))
    data_loader = torch.utils.data.DataLoader(dataset, batch_size=128)

    for data in data_loader:
        data_list.append(data[0])
        prediction_list.append(net(data[0].to(device)).max(1, keepdim=True)[1].cpu())

    data_list = torch.cat(data_list)
    prediction_list = torch.cat(prediction_list)

    plt.scatter(data_list[:, 0], data_list[:, 1], c=prediction_list, s=4)

    test_data, test_label = test_dataset.tensors[0], test_dataset.tensors[1]

    plt.scatter(test_data[:, 0], test_data[:, 1], edgecolor=['w' if i==0 else 'k' for i in test_label[:,1]], s=2)

    plt.axis('off')
    plt.axis('equal')

    plt.savefig('Syn_results/decision_boundary_{}_mode_{}_sample_{}.png'.format(args.dataset, args.mixup_setting, args.num_sample))
    plt.close()

    return


class FC_Net(nn.Module):
    def __init__(self, num_layer=4, data_dim=2, num_class=2):
        super(FC_Net, self).__init__()

        self.num_layer = num_layer
        self.data_dim = data_dim
        self.num_class = num_class

        self.fc1 = torch.nn.Linear(self.data_dim, 64)
        self.fc2 = torch.nn.Linear(64, 128)
        if self.num_layer >= 4:
            self.fc3 = torch.nn.Linear(128, 128)
            self.fc4 = torch.nn.Linear(128, 128)
        if self.num_layer >= 6:
            self.fc5 = torch.nn.Linear(128, 128)
            self.fc6 = torch.nn.Linear(128, 128)
        self.fc7 = torch.nn.Linear(128, self.num_class)

    def latent(self, x):
        return x

    def out(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        if self.num_layer >= 4:
            x = F.relu(self.fc3(x))
            x = F.relu(self.fc4(x))
        if self.num_layer >= 6:
            x = F.relu(self.fc5(x))
            x = F.relu(self.fc6(x))
        x = self.fc7(x)
        return x

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        if self.num_layer >= 4:
            x = F.relu(self.fc3(x))
            x = F.relu(self.fc4(x))
        if self.num_layer >= 6:
            x = F.relu(self.fc5(x))
            x = F.relu(self.fc6(x))
        x = self.fc7(x)
        return x


def init_net(network):
    if network.lower() == 'fc':
        net = FC_Net()
    else:
        print('fill in here')
        exit()

    return net


def adjust_learning_rate(optimizer, epoch):
    lr = args.lr
    if dataset == 'twocircle':
        if epoch >= 20:
            lr /= 10
        if epoch >= 50:
            lr /= 10
    else:
        if epoch >= 60:
            lr /= 10
        if epoch >= 80:
            lr /= 10

    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


@torch.no_grad()
def test(net, test_loader):
    net.eval()

    correct = 0

    for data, label in test_loader:

        data, label = data.to(device), label.to(device)

        output = net(data)

        pred = output.max(1, keepdim=True)[1]
        label = label.max(1, keepdim=True)[1]

        correct += pred.eq(label.view_as(pred)).sum().item()

    accuracy = correct / len(test_loader.dataset)
    print("Testing accuracy: %.3f%%" % (accuracy * 100))

    return accuracy


def train(train_loader, val_loader, train_epoch, mode, lr, dataset, digit_list):

    net = init_net(args.network)
    net.to(device)
    best_acc = -1.0 

    torch.save(net.state_dict(), 'Syn_results/model/' +
               mode + '_{}_{}_{}_{}_best_{}.pkl'.format(args.dataset, args.network, args.dir_num, args.trial, args.validate))

    lossfunc = SoftCrossEntropy().to(device)


    if args.validate == 'cln':
        val_step = 1
    elif args.validate == 'rob':
        val_step = 5
    else:
        print('WRONG VALIDATE')
        exit()

    train_hist = {}
    train_hist['losses'] = []
    train_hist['per_epoch_ptimes'] = []
    train_hist['total_ptime'] = []

    print('Training start!')
    start_time = time.time()

    optimizer = optim.SGD(net.parameters(), lr=lr, momentum=0.9, weight_decay=0.0001)

    show_mixup_data = True

    if mode.endswith('genlabel'):
        lossfunc = SoftCrossEntropy(reduction='none').to(device)
        loss_lam = args.lam

        with torch.no_grad():

            feature_pack, label_pack = train_loader.dataset[:]

            kernel_list, not_none_idx = estimate_kde(feature_pack, label_pack, digit_list, bw=args.bw)

            labels = torch.eye(len(digit_list)).to(device)
            labels = labels[not_none_idx]
            num_class = len(labels)

    for epoch in range(train_epoch):
        losses = []

        epoch_start_time = time.time()
        
        net.train()

        if mode == 'vanilla':
            for data, label in train_loader:

                data, label = data.to(device), label.to(device)

                net.zero_grad()
                output = net(data)
                loss = lossfunc(output, label)

                loss.backward()

                torch.nn.utils.clip_grad_norm_(net.parameters(), 5)

                optimizer.step()

                losses.append(loss.data.item())

        elif mode == 'mixup':
            for data, label in train_loader:

                data, label = data.to(device), label.to(device)

                lam = get_lambda(alpha=1.0)
                mix_data, label_a, label_b = mixup_data(data, label, lam)

                if show_mixup_data:
                    label_to_show = (lam * label_a + (1 - lam) * label_b).cpu()
                    data_to_show = mix_data.cpu()

                    plt.scatter(data_to_show[:, 0], data_to_show[:, 1], c=label_to_show[:, 0], alpha=0.5)
                    plt.legend()

                    plt.savefig('Syn_results/mixup_{}_mode_{}_sample_{}.png'.format(args.dataset, args.mixup_setting, args.num_sample))
                    plt.close()

                    show_mixup_data = False

                net.zero_grad()
                output = net(mix_data)

                loss_func_2 = mixup_criterion(label_a, label_b, lam)
                loss = loss_func_2(lossfunc, output)

                loss.backward()

                torch.nn.utils.clip_grad_norm_(net.parameters(), 5)

                optimizer.step()

                losses.append(loss.data.item())

        elif mode == 'mixup+genlabel':
            for data, label in train_loader:

                data, label = data.to(device), label.to(device)

                net.zero_grad()

                mix_lam = get_lambda(alpha=1.0)
                mid_data, mix_label_1, mix_label_2 = mixup_data(data, label, mix_lam)

                mid_feature_pack = mid_data

                pdf_list = []
                for idx in range(num_class):
                    pdf = kernel_list[idx].logpdf(mid_feature_pack)
                    pdf_list.append(pdf)

                pdf = torch.stack(pdf_list, dim=1).to(device)
                pdf_value, pdf_indices = torch.sort(pdf, dim=-1, descending=True)
                lam = 1 / (1 + torch.exp(pdf_value[:, 1] - pdf_value[:, 0]))

                mid_label_pack = labels.unsqueeze(0).repeat(mid_feature_pack.size(0), 1, 1) \
                    .gather(dim=1, index=pdf_indices[:, 0:2].unsqueeze(-1).repeat(1, 1, labels.size(-1))).squeeze(1)

                label_1 = mid_label_pack[:, 0]
                label_2 = mid_label_pack[:, 1]

                output = net(mid_feature_pack)

                loss_func_2 = mixup_criterion(mix_label_1, mix_label_2, mix_lam)
                loss_func_3 = mixup_criterion(label_1, label_2, lam)

                general_loss = loss_func_2(lossfunc, output).mean()
                gm_loss = loss_func_3(lossfunc, output).mean()

                loss = (1 - loss_lam) * general_loss + loss_lam * gm_loss

                loss.backward()

                torch.nn.utils.clip_grad_norm_(net.parameters(), 5)

                optimizer.step()

                losses.append(loss.data.item())

        else:
            print('Fill in here')
            exit()

        epoch_end_time = time.time()
        per_epoch_ptime = epoch_end_time - epoch_start_time

        print('[%d/%d] - ptime: %.2f, clean loss: %.9f, learning rate: %.6f' % ((epoch + 1),
                                                           train_epoch, per_epoch_ptime,
                                                           torch.mean(torch.FloatTensor(losses)),
                                                           optimizer.param_groups[0]['lr']))

        adjust_learning_rate(optimizer, epoch+1)

        train_hist['losses'].append(torch.mean(torch.FloatTensor(losses)))
        train_hist['per_epoch_ptimes'].append(per_epoch_ptime)

        if (epoch + 1) % val_step == 0:
            net.eval()

            clean_acc = test(net, val_loader)
            print("Validation clean accuracy: %.3f%%" % (clean_acc * 100))

            f = open('log_{}.txt'.format(args.dir_num), "a")
            print(clean_acc * 100, file=f)
            f.close()

        torch.save(net.state_dict(),'Syn_results/model/' +
                   mode + '_{}_{}_{}_{}_best_{}.pkl'.format(args.dataset, args.network, args.dir_num, args.trial, args.validate))

    end_time = time.time()
    total_ptime = end_time - start_time
    train_hist['total_ptime'].append(total_ptime)

    print("Avg one epoch ptime: %.2f, total %d epochs ptime: %.6f" % (
        torch.mean(torch.FloatTensor(train_hist['per_epoch_ptimes'])), train_epoch, total_ptime))
    print("Training finish!... save training results")
    return 0


if __name__ == '__main__':

    args = parser.parse_args()
    seed((int)(42+args.trial))
    print(args)

    batch_size = args.batch_size
    lr = args.lr
    train_epoch = args.epoch
    digit_list = [int(i) for i in range(2)]
    dataset = args.dataset

    root = 'Syn_results/'
    if not os.path.isdir(root):
        os.mkdir(root)
    if not os.path.isdir(root + 'data'):
        os.mkdir(root + 'data')
    if not os.path.isdir(root + 'model'):
        os.mkdir(root + 'model')

    if args.mixup_setting == 0:
        mode = 'vanilla'
    elif args.mixup_setting == 1:
        mode = 'mixup'
    elif args.mixup_setting == 2:
        mode = 'mixup+genlabel'
    elif args.mixup_setting == 3:
        mode = 'mixup_manifold_intrusion'
    elif args.mixup_setting == 4:
        mode = 'kernel_svm'
    else:
        print('Train Mode error')
        exit()

    print('Mode: ' + mode)

    f = open('log_{}.txt'.format(args.dir_num), "a")
    print(args, file=f)
    f.close()

    if mode == 'kernel_svm':
        train_dataset = get_data(dataset, digit_list, train=True)
        val_dataset = get_data(dataset, digit_list, train=False)
        test_dataset = get_data(dataset, digit_list, train=False)

        train_data, train_label = train_dataset.tensors[0], train_dataset.tensors[1]
        val_data, val_label = val_dataset.tensors[0], val_dataset.tensors[1]
        test_data, test_label = test_dataset.tensors[0], test_dataset.tensors[1]

        train_data, train_label = train_data.numpy(), train_label.max(1)[1].numpy()
        val_data, val_label = val_data.numpy(), val_label.max(1)[1].numpy()
        test_data, test_label = test_data.numpy(), test_label.max(1)[1].numpy()

        model = svm.SVC(kernel='rbf', gamma=2).fit(train_data, train_label)

        test_clean_acc = model.score(test_data, test_label)
        print(test_clean_acc)

        prediction_list = []
        data_list = []

        if args.dataset == 'moon':
            scale = torch.Tensor([3, 3]).view(1, 2)
            bias = torch.Tensor([-1, -1.25]).view(1, 2)
        elif args.dataset == 'circle':
            scale = torch.Tensor([3, 3]).view(1, 2)
            bias = torch.Tensor([-1.5, -1.5]).view(1, 2)
        elif args.dataset == 'twocircle':
            scale = torch.Tensor([5, 3]).view(1, 2)
            bias = torch.Tensor([-2.5, -1.5]).view(1, 2)
        data = torch.rand((100000, 2)) * scale + bias

        dataset = torch.utils.data.TensorDataset((data))

        data = dataset.tensors[0].numpy()

        prediction_list = model.predict(data)

        plt.scatter(data[:, 0], data[:, 1], c=prediction_list, s=4)

        test_data, test_label = test_dataset.tensors[0], test_dataset.tensors[1]

        plt.scatter(test_data[:, 0], test_data[:, 1], edgecolor=['w' if i == 0 else 'k' for i in test_label[:, 1]], s=2)

        plt.axis('off')
        plt.axis('equal')

        plt.savefig('Syn_results/decision_boundary_{}_mode_{}_sample_{}.png'.format(args.dataset, args.mixup_setting,
                                                                                    args.num_sample))
        plt.close()
        exit()

    train_dataset = get_data(dataset, digit_list, train=True)
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True, pin_memory=True, num_workers=2)

    val_dataset = get_data(dataset, digit_list, train=False)
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    test_dataset = get_data(dataset, digit_list, train=False)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    if args.train_setting == 1:
        train(train_loader, val_loader, train_epoch, mode, lr, dataset, digit_list)

    best_net = init_net(args.network)
    best_net.load_state_dict(torch.load(
        'Syn_results/model/' + mode + '_{}_{}_{}_{}_best_{}.pkl'.format(args.dataset, args.network, args.dir_num, args.trial, args.validate)))
    best_net.eval()
    best_net.to(device)

    test_data, test_label = extract_dataset(test_dataset)

    test_clean_acc = test(best_net, test_loader)
    print("Test clean accuracy on the best model: %.3f%%" %
          (test_clean_acc * 100))

    f = open('log_{}.txt'.format(args.dir_num), "a")

    decision_boundary(best_net, test_dataset)

    print("Trial:{}, Mode: {}, lr: {}, lam: {}, Bandwidth: {}".format(
        args.trial, mode, args.lr, args.lam, args.bw), file=f)
    print(test_clean_acc * 100, file=f)
    f.close()

    print('Saved results can be found in ' + 'log_{}.txt'.format(args.dir_num))
