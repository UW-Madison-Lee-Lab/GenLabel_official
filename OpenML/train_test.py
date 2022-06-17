
# train_test.py

import pandas as pd
import numpy as np
import numpy.linalg
import pdb
import copy
import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.utils.data import TensorDataset, DataLoader
from sklearn.base import  BaseEstimator, ClassifierMixin
from sklearn.covariance import EmpiricalCovariance
from sklearn.model_selection import cross_val_score

from scipy.stats import multivariate_normal, gaussian_kde
from scipy.special import softmax
from option import opt
from utils import find_nn_label, attack_and_test

def train(mode, arch, X_train, y_train, best_gen_list=None, best_gamma_list=None, best_scheme_list=None, best_lr_list=None, X_test=None, y_test=None):

    best_gen = None
    best_scheme = None
    best_gamma = -1
    best_lr = 0
    best_score = 0
    gamma_list = np.arange(0, 1.01, 0.2)

    if mode in ['vanilla', 'mixup', 'adamixup']:    # cv-param: lr
        if mode == 'vanilla': 		# vanilla training
            clf = Vanilla()
        elif mode == 'mixup':		# mixup
            clf = Mixup(alpha=opt.alpha, nn_label=opt.nn_label, exclude_MI=opt.exclude_MI)      
        elif mode == 'adamixup':   # adamix
            clf = Adamixup()

        # cross-validation on lr
        for lr in opt.lr_list:
            opt.lr = lr
            #pdb.set_trace()
            score_list = cross_val_score(clf, X_train, y_train, cv=5)
            score = score_list.mean() 
            print('lr: {}, score_list: {}, score: {}'.format(lr, score_list, score))
            if score > best_score:
                best_score = score
                best_lr = lr

    elif mode in ['gmlabel', 'kdelabel']:   # mixup + genlabel (GM or KDE), cv-param: lr, gamma
        gen_model = mode[:-5].upper()
        for lr in opt.lr_list:
            print('lr: ', lr)
            opt.lr = lr
            for gamma in gamma_list:
                # define classifier 
                if opt.arch == 'log_reg':
                    clf = GenLabel_LogisticRegression(alpha = opt.alpha, gen_model = gen_model, gamma = gamma)
                elif opt.arch == '3FC':
                    clf = GenLabel_FullyConnected(alpha = opt.alpha, gen_model = gen_model, gamma = gamma)

                # cross-validation on gamma
                score_list = cross_val_score(clf, X_train, y_train, cv=5)
                score = score_list.mean() 
                print('lr: {}, gamma: {}, score_list: {}, score: {}'.format(lr, gamma, score_list, score))				
                #score = cross_val_score(clf, X_train, y_train, cv=5).mean()
                if score > best_score:
                    best_score = score
                    best_gamma = gamma
                    best_lr = lr
        
        #print('best gamma: ', best_gamma)
        best_gamma_list.append(best_gamma)
        if opt.arch == 'log_reg':
            clf = GenLabel_LogisticRegression(alpha = opt.alpha, gen_model = gen_model, gamma = best_gamma)
        elif opt.arch == '3FC':
            clf = GenLabel_FullyConnected(alpha = opt.alpha, gen_model = gen_model, gamma = best_gamma)

    elif mode in ['gmlabel_vanilla']:
        scheme_list = ['gmlabel', 'vanilla']
        for lr in opt.lr_list:
            opt.lr = lr
            print('lr: ', lr)
            for sc in scheme_list:

                for gamma in gamma_list:
                    # define classifier
                    if sc == 'gmlabel': 
                        if opt.arch == 'log_reg':
                            clf = GenLabel_LogisticRegression(alpha = opt.alpha, gen_model = 'GM', gamma = gamma)
                        elif opt.arch == '3FC':
                            clf = GenLabel_FullyConnected(alpha = opt.alpha, gen_model = 'GM', gamma = gamma)
                    elif sc == 'vanilla':
                        clf = Vanilla()
                    else:
                        raise NotImplementedError

                    # cross-validation 
                    score = cross_val_score(clf, X_train, y_train, cv=5).mean()
                    print(sc, gamma, score)
                    if score > best_score:
                        best_score = score
                        best_scheme = sc
                        best_gamma = gamma
                        best_lr = lr

                    if sc == 'vanilla':
                        break


    elif mode in ['1nnlabel_vanilla']:
        scheme_list = ['1nnlabel', 'vanilla']
        for lr in opt.lr_list:
            opt.lr = lr
            print('lr: ', lr)
            for sc in scheme_list:

                # define classifier
                if sc == '1nnlabel': 
                    clf = Mixup(alpha=opt.alpha, nn_label=True)	
                elif sc == 'vanilla':
                    clf = Vanilla()
                else:
                    raise NotImplementedError

                # cross-validation 
                score = cross_val_score(clf, X_train, y_train, cv=5).mean()
                print(lr, sc, score)
                if score > best_score:
                    best_score = score
                    best_scheme = sc
                    best_lr = lr

        best_scheme_list.append(best_scheme)          
        if best_scheme == '1nnlabel':	
            clf = Mixup(alpha=opt.alpha, nn_label=True)	
        elif best_scheme == 'vanilla':
            clf = Vanilla()

    elif mode in ['bestlabel']:
        generative_model_list = ['gm', 'kde']
        for lr in opt.lr_list:
            opt.lr = lr
            for gen in generative_model_list:
                for gamma in gamma_list:
                    # define classifier 
                    if opt.arch == 'log_reg':
                        clf = GenLabel_LogisticRegression(alpha = opt.alpha, gen_model = gen.upper(), gamma = gamma)
                    elif opt.arch == '3FC':
                        clf = GenLabel_FullyConnected(alpha = opt.alpha, gen_model = gen.upper(), gamma = gamma)

                    # cross-validation on gamma
                    score = cross_val_score(clf, X_train, y_train, cv=5).mean()
                    #print(gen, gamma, score)
                    if score > best_score:
                        best_score = score
                        best_gen = gen
                        best_gamma = gamma
                        best_lr = lr

        best_gen_list.append(best_gen)          
        best_gamma_list.append(best_gamma)
        if opt.arch == 'log_reg':
            clf = GenLabel_LogisticRegression(alpha = opt.alpha, gen_model = best_gen.upper(), gamma = best_gamma)
        elif opt.arch == '3FC':
            clf = GenLabel_FullyConnected(alpha = opt.alpha, gen_model = best_gen.upper(), gamma = best_gamma)

    elif mode in ['gen_cls']: # generative classifier
        if opt.arch == 'log_reg':
            clf = Generative_classifier()
        else:
            raise NotImplementedError


    print('mode: {}, best_lr: {}'.format(mode, best_lr))
    best_lr_list.append(best_lr)          
    opt.lr = best_lr  
    clf.fit(X_train, y_train)

    return clf, best_gen_list, best_gamma_list, best_scheme_list, best_lr_list










def get_optimizer(model, opt):
    if opt.optimizer == 'SGD':
        optimizer = torch.optim.SGD(
                                    model.parameters(), 
                                    momentum=0.9, 
                                    lr=opt.lr, 
                                    weight_decay = 0.0001
                                    )
    else:
        optimizer = torch.optim.Adam(
                                    model.parameters(), 
                                    lr=opt.lr, 
                                    weight_decay = 0.0001
                                    )

    return optimizer


def get_optimizer_params(params, opt):
    if opt.optimizer == 'SGD':
        optimizer = torch.optim.SGD(
                                    params, 
                                    momentum=0.9, 
                                    lr=opt.lr, 
                                    weight_decay = 0.0001
                                    )
    else:
        optimizer = torch.optim.Adam(
                                    params, 
                                    lr=opt.lr, 
                                    weight_decay = 0.0001
                                    )

    return optimizer


def get_model(num_of_features, num_of_labels, opt):
    if opt.arch == 'log_reg':
        model = Logistic_Reg_model(num_of_features, num_of_labels)
    elif opt.arch == '3FC':
        model = MLP(num_of_features, num_of_labels)

    return model


class Logistic_Reg_model(torch.nn.Module):
    def __init__(self, num_input_features, num_classes):
        super(Logistic_Reg_model,self).__init__()
        self.layer = torch.nn.Linear(num_input_features, num_classes)
    def forward(self, x):
        y = self.layer(x)
        return y

class MLP(torch.nn.Module):
    def __init__(self, num_input_features, num_classes, num_hidden=128):
        super(MLP,self).__init__()
        self.num_hidden = num_hidden
        self.fc1 = torch.nn.Linear(num_input_features, num_hidden)
        self.fc2 = torch.nn.Linear(num_hidden, num_hidden)
        self.fc3 = torch.nn.Linear(num_hidden, num_classes)

    def forward(self, x, latent=False):
        x = self.fc1(x)
        x = self.fc2(x)
        if latent:
            return x
        y = self.fc3(x)
        return y

    def freeze_feature_extractor(self):
        self.layer.requires_grad = False
        self.layer2.requires_grad = False




class Generative_classifier(BaseEstimator):
    # feature space == input space

    def fit(self, X = None, y = None):

        # 1. estimate p(x|y) using Gaussian mixture model
        m_array, s_array = estimate_gm(X, y) # numpy arrays (mean, cov)
        self.m_array = m_array
        self.s_array = s_array

        # 2. estimate p(y)
        c_array =[]
        num_of_labels = y.max()+1
        for i in range(num_of_labels):
            c_array.append(len(np.where(y == i)[0])/len(y))
        
        #pdb.set_trace()
        self.c_array = c_array
        self.num_of_labels = num_of_labels

    def predict(self, X = None):
        logP_array = np.zeros((X.shape[0], self.num_of_labels))
        for class_idx in range(self.num_of_labels):
            logP_array[:, class_idx] = multivariate_normal(self.m_array[class_idx], 
                                                                                                        self.s_array[class_idx]).logpdf(X)      
            logP_array[:, class_idx] += np.log(self.c_array[class_idx])

        #pdb.set_trace()
        return np.argmax(logP_array, axis=1)

    def score(self, X = None, y = None):
        return (self.predict(X) == y).mean()


class Adamix_lambda_generator(torch.nn.Module):
    def __init__(self, num_input_features):
        super(Adamix_lambda_generator,self).__init__()
        self.layer1 = torch.nn.Linear(num_input_features * 2, 100)
        self.layer2 = torch.nn.Linear(100, 100)
        self.layer3 = torch.nn.Linear(100, 3)
        self.sm = nn.Softmax(dim=1)
    def forward(self, x1, x2):
        y = self.layer1(torch.cat((x1, x2), dim=1))
        y = self.layer2(y)
        y = self.layer3(y)
        y = self.sm(y)
        y = y[:, :2]
        return y

class Adamix_intrusion_classifier(torch.nn.Module):
    def __init__(self, num_input_features):
        super(Adamix_intrusion_classifier,self).__init__()
        self.layer = torch.nn.Linear(num_input_features, 2)
    def forward(self, x):
        y = self.layer(x)
        return y




class Vanilla(BaseEstimator):
    def fit(self, X = None, y = None):
        #print('lr: ', opt.lr) # current lr used for fit & cross_val_score

        num_of_features, num_of_labels = X.shape[1], y.max()+1
        X = torch.from_numpy(X.astype(np.float32))
        y = torch.from_numpy(y.astype(np.long))
        
        my_dataset = TensorDataset(X, y) # create your datset
        my_dataloader = DataLoader(my_dataset, batch_size = 128) # create your dataloader

        model = get_model(num_of_features, num_of_labels, opt)
        #print(model)
        #pdb.set_trace()
        criterion = nn.CrossEntropyLoss()
        optimizer = get_optimizer(model, opt)
        
        loss_sequence = {}
        for epoch in range(opt.num_epochs):
            for batch_ndx, sample in enumerate(my_dataloader):
                X, y = sample
                y_hat = model(X)
                loss = criterion(y_hat, y)
                loss.backward()
                optimizer.step()
                optimizer.zero_grad()
                if epoch % 10 == 0:
                    loss_sequence[epoch] = loss.item()
                    #print('epoch:', epoch, ',loss=', loss.item())

        self.model = model
        self.loss_sequence = loss_sequence

    def predict(self, X = None):
        return self.model(X)

    def score(self, X = None, y = None):
        X = torch.from_numpy(X.astype(np.float32))
        if opt.measure == 'cln':	
            return (np.argmax(self.model(X).detach().numpy(), axis = 1) == y).mean()
        elif opt.measure == 'rob':
            y = torch.from_numpy(y.astype(np.float32))
            rob_acc = attack_and_test(X, y, self.model, opt.Xmin, opt.Xmax, opt.attack_radius_fraction) 
            return rob_acc.detach().numpy().item()
        else:
            raise NotImplementedError

    def get_loss_sequence(self):
        return self.loss_sequence

class Mixup(BaseEstimator):
    def __init__(self, alpha, nn_label=False, exclude_MI=False): #, self_training=False):#, parent_model=None):
        self.alpha = alpha
        self.nn_label = nn_label	
        self.exclude_MI = exclude_MI	
        # if self_training:
        # 	raise NotImplementedError
        # 	#self.parent_model = parent_model
        # 	#self.parent_model.eval()


    def fit(self, X = None, y = None, logistic_label = opt.logistic_label):#, nn_label = opt.nn_label, exclude_MI = opt.exclude_MI):
        X_original, y_original = X, y
        num_of_features, num_of_labels = X.shape[1], y.max()+1
        X = torch.from_numpy(X.astype(np.float32))
        y = torch.from_numpy(y.astype(np.long))
        y = F.one_hot(y).float()
        my_dataset = TensorDataset(X, y) # create your datset
        my_dataloader = DataLoader(my_dataset, batch_size = 128) # create your dataloader

        model = get_model(num_of_features, num_of_labels, opt)
        criterion = nn.CrossEntropyLoss()
        optimizer = get_optimizer(model, opt)
        
        loss_sequence = {}
        for epoch in range(opt.num_epochs):
            for batch_ndx, sample in enumerate(my_dataloader):
                X, y = sample
                
                lam = get_lambda(alpha = self.alpha)
                X_mix, y_a, y_b = mixup_data(X, y, lam)
                y_hat = model(X_mix)

                #pdb.set_trace()
                # variants of mixup
                # if self_training:
                # 	raise NotImplementedError
                # 	#loss = criterion(y_hat, self.parent_model(X_mix))

                #else:
                if logistic_label:
                    sigma_sq = 1/10
                    #print(lam)
                    lam = 1 / (1 + np.exp(-2*(lam-0.5)/sigma_sq))
                    #print(lam)
                elif self.nn_label: # follow the nearest neighbor label
                    #print(y_a)          
                    lam = 1 # use only y_a
                    y_a = find_nn_label(X_mix, X_original, y_original)
                    #print(y_a) 
                if self.exclude_MI:  # exclude manifold intrusion points
                    y_nn = find_nn_label(X_mix, X_original, y_original)
                    y_a_integer, y_b_integer = y_a.argmax(dim=1), y_b.argmax(dim=1)
                    # boolean for NOT suffering manifold intrusion (MI)
                    no_MI_bool = (((y_nn == y_a_integer).int() + (y_nn == y_b_integer).int()) > 0).int() 
                    no_MI_idx = np.where(no_MI_bool.numpy() == 1)[0]
                    y_hat, y_a, y_b = y_hat[no_MI_idx], y_a[no_MI_idx], y_b[no_MI_idx]
                    # exclude some rows of y_hat, y_a, y_b
                loss = lam * criterion(y_hat, y_a) + (1 - lam) * criterion(y_hat, y_b)

                loss.backward()
                optimizer.step()
                optimizer.zero_grad()
                if epoch % 10 == 0:
                    loss_sequence[epoch] = loss.item()
                    #print('epoch:', epoch, ',loss=', loss.item())

        self.model = model
        self.loss_sequence = loss_sequence
        #print(loss_sequence)

    def predict(self, X = None):
        return self.model(X)

    def score(self, X = None, y = None):
        X = torch.from_numpy(X.astype(np.float32))
        if opt.measure == 'cln':	
            return (np.argmax(self.model(X).detach().numpy(), axis = 1) == y).mean()
        elif opt.measure == 'rob':
            y = torch.from_numpy(y.astype(np.float32))
            rob_acc = attack_and_test(X, y, self.model, opt.Xmin, opt.Xmax, opt.attack_radius_fraction) 
            return rob_acc.detach().numpy().item()
        else:
            raise NotImplementedError

    def get_loss_sequence(self):
        return self.loss_sequence

def get_lambda(alpha=1.0):
    if alpha > 0.:
            lam = np.random.beta(alpha, alpha)
    else:
            if np.random.rand() <= 0.5:
                lam = 1.
            else:
                lam = 0.
    return lam

def mixup_data(x, y, lam):
    batch_size = x.size()[0]
    index = torch.randperm(batch_size) # shuffled index
    mixed_x = lam * x + (1 - lam) * x[index, :]
    y_a, y_b = y, y[index]
    return mixed_x, y_a, y_b

class Adamixup(BaseEstimator):
    def __init__(self):
        None
    
    def fit(self, X = None, y = None):
        num_of_features, num_of_labels = X.shape[1], y.max()+1
        X = torch.from_numpy(X.astype(np.float32))
        y = torch.from_numpy(y.astype(np.long))
        y = F.one_hot(y).float()
        my_dataset = TensorDataset(X, y) # create your datset
        my_dataloader = DataLoader(my_dataset, batch_size = 128) # create your dataloader

        # Define models
        model = get_model(num_of_features, num_of_labels, opt)
        ad_generator = Adamix_lambda_generator(num_of_features)
        intrusion_classifier = Adamix_intrusion_classifier(num_of_features)

        criterion = nn.CrossEntropyLoss()
        optimizer = get_optimizer(model, opt)

        loss_sequence = {}
        for epoch in range(opt.num_epochs):
            for batch_ndx, sample in enumerate(my_dataloader):
                X, y = sample
                random_index2 = torch.randperm(X.shape[0]) # shuffled index for X2, y2
                random_index3 = torch.randperm(X.shape[0]) # shuffled index for X3, y3
                #print(random_index2)
                #print(random_index3)

                X1 = X
                X2 = X[random_index2, :]
                X3 = X[random_index3, :]
                y1 = y
                y2 = y[random_index2]
                
                # get lambda from (a,d)
                ad = ad_generator(X1, X2)
                a, d = ad[:,0], ad[:,1]
                e = torch.rand(X.shape[0])
                lam = a + e*d 
                lam = lam.reshape(-1,1)

                X_mix = lam * X1 + (1 - lam) * X2 
                y_hat_mix = model(X_mix)
                y_hat_1 = model(X1)
                y_hat_2 = model(X2)

                intr_hat_mix = intrusion_classifier(X_mix)
                intr_hat_1= intrusion_classifier(X1)
                intr_hat_2 = intrusion_classifier(X2)
                intr_hat_3 = intrusion_classifier(X3)

                # classification loss
                #loss = (lam * criterion(y_hat_mix, y1) + (1 - lam) * criterion(y_hat_mix, y2)).sum()
                loss = criterion(y_hat_mix, (lam.reshape(-1,1)) * y1 + ((1-lam).reshape(-1,1)) * y2)
                loss += criterion(y_hat_1, y1)
                loss += criterion(y_hat_2, y2)
                
                # intrusion detection loss
                
                intr_loss = criterion(intr_hat_mix, torch.tensor([1., 0.]).repeat(X_mix.shape[0], 1)) 
                intr_loss += criterion(intr_hat_1, torch.tensor([0., 1.]).repeat(X1.shape[0], 1))
                intr_loss += criterion(intr_hat_2, torch.tensor([0., 1.]).repeat(X2.shape[0], 1))
                intr_loss += criterion(intr_hat_3, torch.tensor([0., 1.]).repeat(X3.shape[0], 1))
                total_loss = loss + intr_loss

                total_loss.backward()
                optimizer.step()
                optimizer.zero_grad()
            if epoch % 50 == 0:
                loss_sequence[epoch] = total_loss.item()
                # print('epoch:', epoch, ',loss=', loss.item(), intr_loss.item())

        self.model = model
        self.loss_sequence = loss_sequence

    def predict(self, X = None):
        return self.model(X)

    def score(self, X = None, y = None):
        X = torch.from_numpy(X.astype(np.float32))
        if opt.measure == 'cln':	
            return (np.argmax(self.model(X).detach().numpy(), axis = 1) == y).mean()
        elif opt.measure == 'rob':
            y = torch.from_numpy(y.astype(np.float32))
            rob_acc = attack_and_test(X, y, self.model, opt.Xmin, opt.Xmax, opt.attack_radius_fraction) 
            return rob_acc.detach().numpy().item()
        else:
            raise NotImplementedError


    def get_loss_sequence(self):
        return self.loss_sequence

def estimate_gm(X, y, tol=1e-4):
    mu_array, cov_array = [], []
    for i in range(y.max()+1):
        X_slice = X[np.where(y == i)[0], :]
        stat = EmpiricalCovariance().fit(X_slice)
        mu_array.append(stat.location_)
        cov_temp = stat.covariance_ + tol * np.eye(X.shape[1])
        cov_array.append(cov_temp)
    return mu_array, cov_array



def estimate_kde(X, y):
    k_array = []
    for i in range(y.max()+1):
        X_slice = X[np.where(y == i)[0], :]  # print number of samples at each class
        #kernel = gaussian_kde(X_slice.T, bw_method=opt.bw) # gaussian_kde gets 2-dim array (# dim, # data)
        kernel = gaussian_kde_approx(X_slice.T, bw_method=opt.bw) # gaussian_kde gets 2-dim array (# dim, # data)
        k_array.append(kernel)
    return k_array


class gaussian_kde_approx(gaussian_kde):
    # copied from https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.gaussian_kde.html
    # modified to allow the invertibility of covariance matrix

    def _compute_covariance(self):
        """Computes the covariance matrix for each Gaussian kernel using
        covariance_factor().
        """
        self.factor = self.covariance_factor()
        # Cache covariance and inverse covariance of the data
        if not hasattr(self, '_data_inv_cov'):
            self._data_covariance = np.atleast_2d(np.cov(self.dataset, rowvar=1,
                                                                                 bias=False,
                                                                                 aweights=self.weights))
            self._data_covariance += np.eye(self._data_covariance.shape[0]) * 1e-8
            self._data_inv_cov = np.linalg.inv(self._data_covariance)
            #self._data_inv_cov = linalg.inv(self._data_covariance)

        self.covariance = self._data_covariance * self.factor**2
        self.inv_cov = self._data_inv_cov / self.factor**2
        L = np.linalg.cholesky(self.covariance*2*np.pi)
        self.log_det = 2*np.log(np.diag(L)).sum()



class GenLabel_LogisticRegression(BaseEstimator):
    def __init__(self, alpha=opt.alpha, gen_model="GM", gamma=1):
        self.alpha = alpha
        self.gamma = gamma
        self.gen_model = gen_model
    
    def fit(self, X = None, y = None):
        num_of_features, num_of_labels = X.shape[1], y.max()+1
        X_original = X
        y_original = y
        X = torch.from_numpy(X.astype(np.float32))
        y = torch.from_numpy(y.astype(np.long))
        y = F.one_hot(y).float()

        # Learn generative models first
        if self.gen_model == 'GM':
            m_array, s_array = estimate_gm(X_original, y_original) # numpy arrays (mean, cov)
        elif self.gen_model == 'KDE':
            k_array = estimate_kde(X_original, y_original) # arrays of kernels

        # Train
        my_dataset = TensorDataset(X, y) # create your datset
        my_dataloader = DataLoader(my_dataset, batch_size = 128) # create your dataloader

        model = get_model(num_of_features, num_of_labels, opt)
        criterion = nn.CrossEntropyLoss()
        
        optimizer = get_optimizer(model, opt)
        
        loss_sequence = {}
        flag = 0
        for epoch in range(opt.num_epochs):
            for batch_ndx, sample in enumerate(my_dataloader):
                X, y = sample
                
                lam = get_lambda(alpha = self.alpha)
                X_mix, y_a, y_b = mixup_data(X, y, lam) 
                y_hat = model(X_mix)
                
                P_array = np.zeros((X_mix.shape[0], num_of_labels))
                # first try the pdf
                for class_idx in range(num_of_labels):
                    if self.gen_model == 'GM':
                        P_array[:, class_idx] = multivariate_normal(m_array[class_idx], 
                                                                    s_array[class_idx]).pdf(X_mix)
                    elif self.gen_model == 'KDE':
                        P_array[:, class_idx] = k_array[class_idx].pdf(X_mix.T)

                # use logpdf for the cases when P_array = all-zero
                if (P_array.sum(axis=1) == np.zeros(X_mix.shape[0])).any():
                    if flag == 0:
                        print('Using logpdf instead')
                        flag = 1
                    logP_array = np.zeros((X_mix.shape[0], num_of_labels))
                    for class_idx in range(num_of_labels):
                        if self.gen_model == 'GM':
                            logP_array[:, class_idx] = multivariate_normal(m_array[class_idx], 
                                                                            s_array[class_idx]).logpdf(X_mix)
                        elif self.gen_model == 'KDE':
                            logP_array[:, class_idx] = k_array[class_idx].logpdf(X_mix.T)
                    logP_softmax = softmax(logP_array, axis=1)
                else:
                    logP_softmax = P_array/(P_array.sum(axis=1)).reshape(X_mix.shape[0], 1) 


                loss = self.gamma * criterion(y_hat, torch.from_numpy(logP_softmax)) + \
                            (1-self.gamma) * criterion(y_hat, lam * y_a + (1-lam) * y_b)
                loss.backward()
                optimizer.step()
                optimizer.zero_grad()
                if epoch % 10 == 0:
                    loss_sequence[epoch] = loss.item()
                    # print('epoch:', epoch, ',loss=', loss.item())

        self.model = model
        self.loss_sequence = loss_sequence

    def predict(self, X = None):
        return self.model(X)

    def score(self, X = None, y = None):
        X = torch.from_numpy(X.astype(np.float32))
        if opt.measure == 'cln':	
            return (np.argmax(self.model(X).detach().numpy(), axis = 1) == y).mean()
        elif opt.measure == 'rob':
            y = torch.from_numpy(y.astype(np.float32))
            rob_acc = attack_and_test(X, y, self.model, opt.Xmin, opt.Xmax, opt.attack_radius_fraction) 
            return rob_acc.detach().numpy().item()
        else:
            raise NotImplementedError

    def get_loss_sequence(self):
        return self.loss_sequence        


def set_parameter_requires_grad(model):
    for param in model.parameters():
        param.requires_grad = False


class GenLabel_FullyConnected(BaseEstimator):
    def __init__(self, alpha=opt.alpha, gen_model="GM", gamma=1):
        self.alpha = alpha
        self.gamma = gamma
        self.gen_model = gen_model
    
    def fit(self, X = None, y = None):
        num_of_features, num_of_labels = X.shape[1], y.max()+1
        X_original = X
        y_original = y
        X = torch.from_numpy(X.astype(np.float32))
        y = torch.from_numpy(y.astype(np.long))
        X_total = X
        y_total = y
        y_total_onehot = F.one_hot(y).float()


        # 1. use vanilla training for warmup (half of the total epochs)		
        warmup_dataset = TensorDataset(X_total, y_total) # create your datset
        warmup_dataloader = DataLoader(warmup_dataset, batch_size = 128) # create your dataloader

        model = get_model(num_of_features, num_of_labels, opt)
        criterion = nn.CrossEntropyLoss()
        optimizer = get_optimizer(model, opt)
        
        loss_sequence = {}
        half_num_epochs = 20 #int(opt.num_epochs/2)
        for epoch in range(half_num_epochs):
            for batch_ndx, sample in enumerate(warmup_dataloader):
                X, y = sample
                y_hat = model(X)
                loss = criterion(y_hat, y)
                loss.backward()
                optimizer.step()
                optimizer.zero_grad()
                if epoch % 10 == 0:
                    loss_sequence[epoch] = loss.item()
                    #print('epoch:', epoch, ',loss=', loss.item())


        # 2. Learn the generative model for the latent feature
        Z_total = model(X_total, latent=True).detach().numpy() # hidden representation 

        if self.gen_model == 'GM':
            m_array, s_array = estimate_gm(Z_total, y_original) # numpy arrays (mean, cov)
        elif self.gen_model == 'KDE':
            k_array = estimate_kde(Z_total, y_original) # arrays of kernels


        # 3. Fine-tune the last layer only (for half of the total epochs)
        my_dataset = TensorDataset(X_total, y_total_onehot) # create your datset
        my_dataloader = DataLoader(my_dataset, batch_size = 128) # create your dataloader

        criterion = nn.CrossEntropyLoss()

        # freeze the feature extractor part
        # use https://pytorch.org/tutorials/beginner/finetuning_torchvision_models_tutorial.html
        set_parameter_requires_grad(model)
        model.fc3 = nn.Linear(model.num_hidden, num_of_labels) # replace to new layer

        params_to_update = []
        for name,param in model.named_parameters():
            if param.requires_grad == True:
                params_to_update.append(param)
                #print("\t",name)
        optimizer = get_optimizer_params(params_to_update, opt) # set the parameters to finetune
        
        # sanity check
        Z_total_re = model(X_total, latent=True).detach().numpy() # hidden representation 
        assert((Z_total == Z_total_re).all()) 
        

        flag = 0
        for epoch in range(half_num_epochs, opt.num_epochs):
            for batch_ndx, sample in enumerate(my_dataloader):
                X, y = sample
                
                lam = get_lambda(alpha = self.alpha)
                X_mix, y_a, y_b = mixup_data(X, y, lam) 
                y_hat = model(X_mix)
                
                #get GenLabel (logP_softmax)

                # P_array = np.zeros((X_mix.shape[0], num_of_labels))
                # # first try the pdf
                # for class_idx in range(num_of_labels):
                # 	if self.gen_model == 'GM':
                # 		P_array[:, class_idx] = multivariate_normal(m_array[class_idx], 
                # 																								 s_array[class_idx]).pdf(model(X_mix, latent=True).detach())
                # 	elif self.gen_model == 'KDE':
                # 		P_array[:, class_idx] = k_array[class_idx].pdf(model(X_mix, latent=True).detach().T)

                #use logpdf! # for the cases when P_array = all-zero
                if True: #(P_array.sum(axis=1) == np.zeros(X_mix.shape[0])).any():
                    if flag == 0:
                        #print('Using logpdf instead')
                        flag = 1
                    logP_array = np.zeros((X_mix.shape[0], num_of_labels))
                    for class_idx in range(num_of_labels):
                        if self.gen_model == 'GM':
                            logP_array[:, class_idx] = multivariate_normal(m_array[class_idx], 
                                                                            s_array[class_idx]).logpdf(model(X_mix, latent=True).detach())
                        elif self.gen_model == 'KDE':
                            logP_array[:, class_idx] = k_array[class_idx].logpdf(model(X_mix, latent=True).detach().T)
                    logP_softmax = softmax(logP_array, axis=1)
                #else:
                #	logP_softmax = P_array/(P_array.sum(axis=1)).reshape(X_mix.shape[0], 1) 


                loss = self.gamma * criterion(y_hat, torch.from_numpy(logP_softmax)) + \
                            (1-self.gamma) * criterion(y_hat, lam * y_a + (1-lam) * y_b)
                loss.backward()
                optimizer.step()
                optimizer.zero_grad()
                if epoch % 10 == 0:
                    loss_sequence[epoch] = loss.item()

        Z_total_re2 = model(X_total, latent=True).detach().numpy() # hidden representation 
        assert((Z_total == Z_total_re2).all())
        #print('passed sanity check')

        self.model = model
        self.loss_sequence = loss_sequence

    def predict(self, X = None):
        return self.model(X)

    def score(self, X = None, y = None):
        X = torch.from_numpy(X.astype(np.float32))
        if opt.measure == 'cln':	
            return (np.argmax(self.model(X).detach().numpy(), axis = 1) == y).mean()
        elif opt.measure == 'rob':
            y = torch.from_numpy(y.astype(np.float32))
            rob_acc = attack_and_test(X, y, self.model, opt.Xmin, opt.Xmax, opt.attack_radius_fraction) 
            return rob_acc.detach().numpy().item()
        else:
            raise NotImplementedError

    def get_loss_sequence(self):
        return self.loss_sequence        



