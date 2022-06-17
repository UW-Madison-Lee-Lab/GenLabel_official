#!/usr/bin/env python
# -*- coding: UTF-8 -*-

import os
import time
import argparse
import torch
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
from utils import SoftCrossEntropy, extract_dataset, seed
from utils import get_lambda, mixup_data, mixup_criterion

parser = argparse.ArgumentParser(description='2d cube in 3-layer fc net')
parser.add_argument('-n', '--network', default='fc', type=str,
                    help='network used in training')
parser.add_argument('-t', '--train_setting', default=1, type=int,
                    help='mode for training setting, 0: testing, 1: training')
parser.add_argument('-m', '--mixup_setting', default=3, type=int,
                    help='mode for mixup setting, 0: vanilla training, 1: mixup, 2: mixup+genlabel')
parser.add_argument('-b', '--batch_size', default=128, type=int,
                    help='batch_size')
parser.add_argument('-l', '--lr', default=0.1, type=float,
                    help='learning rate')
parser.add_argument('-e', '--epoch', default=40, type=int,
                    help='number of training epochs')
parser.add_argument('--trial', default=1, type=int,
                    help='trial number (1,2, ...), will change random seed')
parser.add_argument('--dir_num', default=1, type=int,
                    help='directory number, suffix for log & model')
parser.add_argument('--dataset', default='2dcube', type=str,
                    help='name of dataset to use')
parser.add_argument('--lam', default=1, type=float,
                    help='loss ratio')
parser.add_argument('--finetune', default=0, type=int,
                    help='finetune epoch')
parser.add_argument('-v', '--validate', default='cln', type=str,
                    help='how to validate, |cln|')
parser.add_argument('--num_sample', default=20, type=int,
                    help='number of samples per class')

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def generate_data(num_sample):

    train_data_1 = torch.Tensor(num_sample, 2).uniform_(-1, 1) - torch.Tensor([1, 0])
    train_data_2 = torch.Tensor(num_sample, 2).uniform_(-1, 1) + torch.Tensor([1, 0])
    train_label_1 = torch.zeros(num_sample)
    train_label_2 = torch.ones(num_sample)
    train_data = torch.cat([train_data_1, train_data_2])
    train_label = torch.cat([train_label_1, train_label_2]).long()

    return train_data, train_label


def get_data(dataset, digit_list, train=True):

    onehot = torch.eye(2)

    if train:
        num_sample = args.num_sample
    else:
        num_sample = args.num_sample

    train_data, train_label = generate_data(num_sample)

    plt.scatter(train_data[:, 0], train_data[:, 1], c=train_label, alpha=0.5)

    if train:
        plt.savefig('2dcube_results/train_data_mode_{}_sample_{}.png'.format(args.mixup_setting, args.num_sample))
    else:
        plt.savefig('2dcube_results/test_data_mode_{}_sample_{}.png'.format(args.mixup_setting, args.num_sample))
    plt.close()

    train_label = onehot[train_label]

    dataset = torch.utils.data.TensorDataset(train_data, train_label)

    return dataset


def decision_boundary(net, data_loader):

    net.eval()

    prediction_list = []
    data_list = []
    for data, label in data_loader:
        data_list.append(data)
        prediction_list.append(net(data.to(device)).max(1, keepdim=True)[1].cpu())

    data_list = torch.cat(data_list)
    prediction_list = torch.cat(prediction_list)

    plt.scatter(data_list[:, 0], data_list[:, 1], c=prediction_list, alpha=0.2)

    plt.savefig('2dcube_results/decision_boundary_mode_{}_sample_{}.png'.format(args.mixup_setting, args.num_sample))
    plt.close()

    return


class FC_Net(nn.Module):

    def __init__(self):
        super().__init__()
        n_cls = 2
        self.fc1 = nn.Linear(2, 64)
        self.fc2 = nn.Linear(64, 128)
        self.fc3 = nn.Linear(128, n_cls)

    def forward(self, input):
        x = F.relu(self.fc1(input))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)

        return x

    def latent(self, input):
        return input

    def out(self, input):

        x = F.relu(self.fc1(input))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)

        return x


def init_net(network):
    if network.lower() == 'fc':
        net = FC_Net()
    else:
        print('fill in here')
        exit()

    return net


def update_Gaussian(feature, mean_stored, cov_stored, label, digit_list):
    mean_pack, cov_pack = [], []
    not_none_idx = []

    if mean_stored is None:

        for idx, digit in enumerate(digit_list):
            feature_selected = feature[torch.max(label, dim=1)[1] == digit]

            if feature_selected.size(0) == 0:
                mean = torch.zeros(0, feature.size(1)).to(device)
                covariance = torch.zeros(0).to(device)

            else:
                mean = torch.mean(feature_selected, dim=0, keepdim=True).detach()
                covariance = torch.ones(1).to(device)
                not_none_idx.append(idx)

            mean_pack.append(mean)
            cov_pack.append(covariance)

    else:

        for idx, digit in enumerate(digit_list):
            feature_selected = feature[torch.max(label, dim=1)[1] == digit]

            if mean_stored[idx].size(0) == 0:

                if feature_selected.size(0) == 0:
                    mean = torch.zeros(0, feature.size(1)).to(device)
                    covariance = torch.zeros(0).to(device)

                else:
                    mean = torch.mean(feature_selected, dim=0, keepdim=True).detach()
                    covariance = torch.ones(1).to(device)
                    not_none_idx.append(idx)

            else:

                not_none_idx.append(idx)

                if feature_selected.size(0) == 0:
                    mean = mean_stored[idx]
                    covariance = cov_stored[idx]

                else:
                    if idx == 0:
                        mean = torch.Tensor([-1, 0]).view(1, 2).to(device)
                    else:
                        mean = torch.Tensor([1, 0]).view(1, 2).to(device)

                    covariance = torch.ones(1).to(device)

            mean_pack.append(mean)
            cov_pack.append(covariance)

    return mean_pack, cov_pack, not_none_idx


def generate_label(mid_feature_pack, mean_pack, cov_pack, digit_list, not_none_idx):

    labels = torch.eye(len(digit_list)).to(device)

    labels = labels[not_none_idx]

    num_class = len(labels)

    mean_pack = torch.cat(mean_pack)
    cov_pack = torch.cat(cov_pack)
    cov_inv_pack = 1 / cov_pack
    log_det_pack = torch.log(cov_pack) * mean_pack.size(-1)

    mean = mean_pack.unsqueeze(0).repeat(mid_feature_pack.size(0), 1, 1)
    cov_inv = cov_inv_pack.view(1, -1).repeat(mid_feature_pack.size(0), 1)
    log_det = log_det_pack.view(1, -1).repeat(mid_feature_pack.size(0), 1)
    mid_feature = mid_feature_pack.detach().unsqueeze(1).repeat(1, num_class, 1)

    pdf = -0.5 * ((mid_feature - mean) ** 2).sum(dim=-1).squeeze() * cov_inv - 0.5 * log_det
    pdf_value, pdf_indices = torch.sort(pdf, dim=-1, descending=True)
    lam = 1 / (1 + torch.exp(pdf_value[:, 1] - pdf_value[:, 0]))

    mid_label_pack = labels.unsqueeze(0).repeat(mid_feature_pack.size(0), 1, 1) \
        .gather(dim=1, index=pdf_indices[:, 0:2].unsqueeze(-1).repeat(1, 1, labels.size(-1))).squeeze(1)

    label_1 = mid_label_pack[:, 0]
    label_2 = mid_label_pack[:, 1]

    return lam, label_1, label_2


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

    torch.save(net.state_dict(), '2dcube_results/model/' +
               mode + '_{}_{}_{}_best_{}.pkl'.format(args.network, args.dir_num, args.trial, args.validate))

    lossfunc = SoftCrossEntropy().to(device)

    optimizer = optim.SGD(net.parameters(), lr=lr, momentum=0.9, weight_decay=0.0001)

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

    show_mixup_data = True

    if mode.endswith('genlabel'):
        mean_stored = None
        cov_stored = None
        lossfunc = SoftCrossEntropy(reduction='none').to(device)

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

                    plt.savefig('2dcube_results/mixup_mode_{}_sample_{}.png'.format(args.mixup_setting, args.num_sample))
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

            if epoch < args.finetune:
                loss_lam = 0
                if epoch == 0:
                    print('warm up epochs')
            else:
                loss_lam = args.lam

            for data, label in train_loader:

                data, label = data.to(device), label.to(device)

                net.zero_grad()

                with torch.no_grad():
                    feature = net.latent(data)

                mix_lam = get_lambda(alpha=1.0)
                mid_data, mix_label_1, mix_label_2 = mixup_data(data, label, mix_lam)

                mid_feature_pack = net.latent(mid_data)

                mean_pack, cov_pack, not_none_idx = update_Gaussian(feature, mean_stored, cov_stored, label, digit_list)

                mean_stored = mean_pack
                cov_stored = cov_pack

                lam, label_1, label_2 = generate_label(mid_feature_pack, mean_pack, cov_pack, digit_list, not_none_idx)

                output = net.out(mid_feature_pack)

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

        train_hist['losses'].append(torch.mean(torch.FloatTensor(losses)))
        train_hist['per_epoch_ptimes'].append(per_epoch_ptime)

        if (epoch + 1) % val_step == 0:
            net.eval()

            clean_acc = test(net, val_loader)
            print("Validation clean accuracy: %.3f%%" % (clean_acc * 100))

            f = open('log_{}.txt'.format(args.dir_num), "a")
            print(clean_acc * 100, file=f)
            f.close()

        torch.save(net.state_dict(), '2dcube_results/model/' +
                   mode + '_{}_{}_{}_best_{}.pkl'.format(args.network, args.dir_num, args.trial, args.validate))

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

    root = '2dcube_results/'
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
    else:
        print('Train Mode error')
        exit()

    print('Mode: ' + mode)

    f = open('log_{}.txt'.format(args.dir_num), "a")
    print(args, file=f)
    f.close()

    train_dataset = get_data(dataset, digit_list, train=True)
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True, pin_memory=True, num_workers=2)

    test_dataset = get_data(dataset, digit_list, train=False)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    val_dataset, val_loader = test_dataset, test_loader

    if args.train_setting == 1:
        train(train_loader, val_loader, train_epoch, mode, lr, dataset, digit_list)

    best_net = init_net(args.network)
    best_net.load_state_dict(torch.load(
        '2dcube_results/model/' + mode + '_{}_{}_{}_best_{}.pkl'.format(args.network, args.dir_num, args.trial, args.validate)))
    best_net.eval()
    best_net.to(device)

    args.num_sample = 10000
    test_dataset = get_data(dataset, digit_list, train=False)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    test_data, test_label = extract_dataset(test_dataset)

    test_clean_acc = test(best_net, test_loader)
    print("Test clean accuracy on the best model: %.3f%%" %
          (test_clean_acc * 100))

    f = open('log_{}.txt'.format(args.dir_num), "a")

    print("Trial:{}, Mode: {}, lr: {}, lam: {}".format(
        args.trial, mode, args.lr, args.lam), file=f)
    print(test_clean_acc * 100, file=f)
    f.close()

    print('Saved results can be found in ' + 'log_{}.txt'.format(args.dir_num))
