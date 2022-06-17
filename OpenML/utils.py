import torch
import random
import numpy as np
import os
import pdb
import foolbox as fb
from option import opt

def set_seed(seed):
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    #print("Seeded everything: {}".format(seed))


def load_data():
    with open('data/openml_train.npy', 'rb') as f:
        openml_datasets_train = np.load(f, allow_pickle=True)
        openml_datasets_train = openml_datasets_train.item() # access 0d array
    with open('data/openml_test.npy', 'rb') as f:
        openml_datasets_test = np.load(f, allow_pickle=True)
        openml_datasets_test = openml_datasets_test.item() # access 0d array

    return openml_datasets_train, openml_datasets_test


def init_result_dict(scheme_list):
    result_dict = {'did': []}
    for scheme in scheme_list:	
        result_dict[scheme+' raw'] = []
        result_dict[scheme+' mean'] = []
        result_dict[scheme+' std'] = []
        result_dict[scheme+' best lr'] = []

        if scheme in ['gmlabel', 'kdelabel']:
            result_dict[scheme+' best gamma'] = []
        elif scheme in ['bestlabel']:
            result_dict[scheme+' best gen'] = []
            result_dict[scheme+' best gamma'] = []
        elif scheme in ['gmlabel_vanilla']:
            result_dict[scheme+' best scheme'] = []
            result_dict[scheme+' best gamma'] = []
        elif scheme in ['1nnlabel_vanilla']:
            result_dict[scheme+' best scheme'] = []


    print(result_dict)

    return result_dict


def save_result_dict(did, scheme, result_dict, tmp_result_list):

    # print the 5 trials in tmp_result_list
    print(did, scheme, tmp_result_list)

    # save avg & std in the result_dict
    ls = np.array(tmp_result_list)
    result_dict[scheme+' raw'].append(ls)
    result_dict[scheme+' mean'].append(ls.mean())
    result_dict[scheme+' std'].append(ls.std())

    return result_dict


def find_nn_label(X_mix, X_original, y_original):

    num_of_labels = y_original.max()+1
    num_batch_data = X_mix.shape[0]
    num_total_data = X_original.shape[0]

    y_nn = torch.ones(num_batch_data) * -1
    #pdb.set_trace()

    # find nearest neighbor 
    for i in range(num_batch_data):
        curr_X = X_mix[i].reshape(1, -1).repeat(num_total_data, 1)
        nn_idx = torch.norm(curr_X - X_original, dim=1).argmin()
        y_nn[i] = y_original[nn_idx]

    assert((y_nn < num_of_labels).all())
    assert((y_nn >= 0).all())

    return y_nn.long()

def get_did_list(opt):

    # all datasets with num_data < 2000, num_feat < 20
    if opt.partition == 1:
        did_list = [18, 36, 37, 48, 53, 54, 187, 307, 446, 467, 468, 472, 476, 682, 683, 717, 720, 721, 726]
    elif opt.partition == 2:
        did_list = [729, 730, 731, 736, 737, 740, 743, 744, 745, 749, 750, 751, 754, 755, 756, 759, 762, 763, 770, 772]
    elif opt.partition == 3:
        did_list = [774, 776, 777, 778, 780, 783, 787, 789, 792, 793, 795, 799, 804, 808, 813, 815, 817, 818, 820, 824]
    elif opt.partition == 4:
        did_list = [825, 827, 829, 830, 836, 841, 845, 853, 855, 857, 859, 862, 863, 864, 869, 870, 871, 874, 878, 880]
    elif opt.partition == 5:
        did_list = [882, 884, 885, 886, 891, 892, 893, 894, 900, 910, 911, 912, 913, 915, 916, 925]
    elif opt.partition == 6:
        did_list = [927, 929, 931, 935, 936, 938, 943, 945, 955, 973, 974, 988, 996, 997, 1005, 1006, 1011, 1048, 1073]
    elif opt.partition == 7:
        did_list = [1167, 1413, 1462, 1465, 1482, 1498, 1499, 1552, 1553, 1557]
    elif opt.partition == 8:
        did_list = [40496, 40710, 40981, 40984, 40999, 41004, 41005, 41007, 41143]

    # balanced datasets with num_data < 500, num_feat < 20
    elif opt.partition == 11:
        did_list = [446, 468]
    elif opt.partition == 12:
        did_list = [476, 683]
    elif opt.partition == 13:
        did_list = [755, 759]
    elif opt.partition == 14:
        did_list = [763, 776]
    elif opt.partition == 15:
        did_list = [880, 894]
    elif opt.partition == 16:
        did_list = [929, 1413, 1499]

    elif opt.partition == 20:
        did_list = [880, 894, 929, 1413, 1499]

    elif opt.partition == 21:
        did_list = [446, 468, 476]
    elif opt.partition == 22:
        did_list = [683, 755, 759]
    elif opt.partition == 23:
        did_list = [763, 776, 880]
    elif opt.partition == 24:
        did_list = [894, 929, 1413, 1499]

    
    else:
        raise NotImplementedError

    return did_list


def attack_and_test(X, y, model, Xmin, Xmax, attack_radius_fraction):

    model.eval()

    # load model
    bounds = (Xmin, Xmax)
    fmodel = fb.PyTorchModel(model, bounds=bounds)

    # check the clean accuracy
    #print(X, y)
    #cln_acc = fb.utils.accuracy(fmodel, X, y)
    #print('clean accuracy: ', cln_acc)

    # attack model
    attack = fb.attacks.BoundaryAttack()
    epsilons = (Xmax - Xmin) * attack_radius_fraction 
    raw, clipped, is_adv = attack(fmodel, X, y, epsilons=epsilons)
    rob_acc = 1 - is_adv.float().mean(axis=-1)
    #print(epsilons, rob_acc)

    model.train()


    return rob_acc

