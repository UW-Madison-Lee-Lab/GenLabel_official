#opt.py

import argparse

def parse_arguments():
    parser = argparse.ArgumentParser(description='GenLabel')
    parser.add_argument('--num_trial', type=int, default=5, help='number of random trials')
    parser.add_argument('--short', '-s', type=int, default=0) 	
    parser.add_argument('--did', type=int, default=None) 	
    parser.add_argument('--partition', '-p', type=int, default=0) 	
    parser.add_argument('--arch', '-a', type=str, default='log_reg',
                                    help='architecture: log_reg | 3FC') # logistic regression / 3-layer FC NN (2 hidden layers with 128 neurons)
    parser.add_argument('--measure', type=str, default='cln',
                                            help='test measure: cln | rob') # clean acc / robust acc


    parser.add_argument('--optimizer', '-o', type=str, default='adam',
                                    help='optimizer: sgd | adam')
    parser.add_argument('--lr', type=float, default=-1,
                                    help='learning rate: 1e-1, 1e-2, 1e-3, 1e-4')
    parser.add_argument('--lr_list', default=[0.1, 0.01, 0.001, 0.0001],
                                    help='learning rate: 1e-1, 1e-2, 1e-3, 1e-4')
    parser.add_argument('--alpha', type=float, default=1,
                                    help='alpha for mixup+')
    parser.add_argument('--num_epochs', type=int, default=100)
    parser.add_argument('--attack_radius_fraction', type=float, default=0.1)
    parser.add_argument('--attack_step', type=int, default=1000)

    parser.add_argument('--mode', '-m', default='all', nargs='+')
    parser.add_argument('--num_feat_category', '-f', type=int, default=0) 
    #f=0: num_feat <=20, f=1: num_feat > 20, f=-1: all datasets
    parser.add_argument('--bw', '-b', type=float, default=None) 
    # bandwidth for KDE (None: Use Scott's method)

    parser.add_argument('--logistic_label', type=int, default=0) 
    parser.add_argument('--nn_label', type=int, default=0) 
    parser.add_argument('--exclude_MI', type=int, default=0) 

    #parser.add_argument('--gen_model', type=str, default='GM') # used for test_gen.py 

    #parser.add_argument('--toy_test', '-t', type=int, default=0)
    opt = parser.parse_args()

    return opt


def run_args():
    global opt
    opt = parse_arguments()
    print(opt)

run_args()
