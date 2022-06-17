
"""main.py

"""
import pandas as pd
import numpy as np
import numpy.linalg
import torch
import torch.nn as nn
import torch.nn.functional as F
import argparse
import time
import os

from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.linear_model import LogisticRegression

from train_test import *
from utils import set_seed, load_data, init_result_dict, save_result_dict, get_did_list
from option import opt


def main():

    # load data
    openml_datasets_train, openml_datasets_test = load_data()

    # initialize result_dict, did_list
    if opt.mode == 'all':
        scheme_list = ['vanilla', 'mixup', 'adamixup',
                       'gmlabel', 'kdelabel', 'bestlabel']
    else:
        scheme_list = opt.mode
    print('scheme list: ', scheme_list)
    result_dict = init_result_dict(scheme_list)
    valid_partition_number = [*range(1,9), *range(11,15), 20, *range(21,25)]
    if opt.partition in valid_partition_number:
        did_list = get_did_list(opt)

    # train & test for each dataset
    for did in openml_datasets_train.keys():

        X_train, y_train = openml_datasets_train[did]["X"], openml_datasets_train[did]["y"]
        X_test, y_test = openml_datasets_test[did]["X"], openml_datasets_test[did]["y"]
        if opt.did and did != opt.did:
            continue
        if X_train.shape[1] > 20:  # run only for num_feat <= 20
            continue
        if opt.partition in valid_partition_number and did not in did_list:
            continue
        opt.Xmin, opt.Xmax = X_train.min(), X_train.max()
        if opt.measure == 'rob':
            # min, max, attack_radius
            print(opt.Xmin, opt.Xmax, (opt.Xmax - opt.Xmin)*opt.attack_radius_fraction)

        result_dict['did'].append(did)
        print('did: ', did)
        start_time = time.time()
        for scheme in scheme_list:
            tmp_result_list, best_gen_list, best_gamma_list, best_scheme_list, best_lr_list = [], [], [], [], []  # temporary result list (for saving 5 trials)

            for trial_iter in range(opt.num_trial):
                set_seed(trial_iter)
                print('trial iter: ', trial_iter)
                clf, best_gen_list, best_gamma_list, best_scheme_list, best_lr_list = train(
                    scheme, opt.arch, X_train, y_train, best_gen_list, best_gamma_list, best_scheme_list, best_lr_list, X_test, y_test)
                #print(best_gen_list, best_gamma_list, best_lr_list)
                tmp_result_list.append(clf.score(X_test, y_test))
            # store mean/std in the list
            result_dict = save_result_dict(did, scheme, result_dict, tmp_result_list)

            result_dict[scheme+' best lr'].append(best_lr_list)
            if scheme in ['gmlabel', 'kdelabel']:
                result_dict[scheme+' best gamma'].append(best_gamma_list)
            elif scheme in ['bestlabel']:
                result_dict[scheme+' best gen'].append(best_gen_list)
                result_dict[scheme+' best gamma'].append(best_gamma_list)
            elif scheme in ['gmlabel_vanilla']:
                result_dict[scheme+' best scheme'].append(best_scheme_list)
                result_dict[scheme+' best gamma'].append(best_gamma_list)
            elif scheme in ['1nnlabel_vanilla']:
                result_dict[scheme+' best scheme'].append(best_scheme_list)


        end_time = time.time()
        print('ptime: {:.2f}'.format(end_time - start_time))

        # print(result_dict)
        # save result
        csv_root = 'csv/'
        if not os.path.exists(csv_root):
            os.makedirs(csv_root)
        df = pd.DataFrame.from_dict(result_dict)
        filename = '{}_{}_result_{}_best_lr_{}_partition_{}_alpha_{}_cat_{}_log_{}_nn_{}_eMI_{}_eps_{}.csv'.format \
        (opt.arch, opt.measure, opt.optimizer,opt.mode, opt.partition, opt.alpha, opt.num_feat_category, \
            opt.logistic_label, opt.nn_label, opt.exclude_MI, opt.attack_radius_fraction).replace \
        ("[", "_").replace("]", "_").replace("'", "_").replace(", ", "_").replace("__", "_").replace("__", "_")
        df.to_csv(csv_root + filename, index=False)


if __name__ == "__main__":
    start_total_time = time.time()

    main()

    end_total_time = time.time()
    print('Total ptime: {:.2f}'.format(end_total_time - start_total_time))
