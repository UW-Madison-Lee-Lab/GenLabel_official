
# train_test.py

import pandas as pd
import numpy as np
import numpy.linalg
import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.utils.data import TensorDataset, DataLoader
from sklearn.base import  BaseEstimator, ClassifierMixin
from sklearn.covariance import EmpiricalCovariance

from scipy.stats import multivariate_normal
#from scipy.special import softmax


def train(mode):

  if mode == 'sklearn_lr':    # sklearn official logistic regression
    clf = LogisticRegression(random_state=0)
  elif mode == 'custom_lr':   # custom logistic regression 
    clf = Custom_LogisticRegression()
  elif mode == 'mixup_lr':   # mixup
    clf = Mixup_LogisticRegression(alpha = 1)
  elif mode == 'adamixup_lr':   # adamix
    clf = Adamixup_LogisticRegression()
  elif mode == 'gmlabel_lr':   # mixup + genlabel (GM)
    clf = GenLabel_LogisticRegression(alpha = 1, gen_model = "GM", gamma = 0.5)

  clf.fit(X_train, y_train)

  return clf



class Logistic_Reg_model(torch.nn.Module):
    def __init__(self, num_input_features, num_classes):
      super(Logistic_Reg_model,self).__init__()
      self.layer = torch.nn.Linear(num_input_features, num_classes)
    def forward(self, x):
      y = self.layer(x)
      return y

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




class Custom_LogisticRegression(ClassifierMixin):
    def fit(self, X = None, y = None):
        num_of_features, num_of_labels = X.shape[1], y.max()+1
        X = torch.from_numpy(X.astype(np.float32))
        y = torch.from_numpy(y.astype(np.long))
        
        my_dataset = TensorDataset(X, y) # create your datset
        my_dataloader = DataLoader(my_dataset, batch_size = 128) # create your dataloader

        model = Logistic_Reg_model(num_of_features, num_of_labels)
        criterion = nn.CrossEntropyLoss()
        
        optimizer = torch.optim.SGD(
                                    model.parameters(), 
                                    momentum=0.9, 
                                    lr=0.02, 
                                    weight_decay = 0.0001
                                    )
        
        loss_sequence = {}
        number_of_epochs = 100
        for epoch in range(number_of_epochs):
          for batch_ndx, sample in enumerate(my_dataloader):
            X, y = sample
            y_hat = model(X)
            loss = criterion(y_hat, y)
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
        return (np.argmax(self.model(X).detach().numpy(), axis = 1) == y).mean()

    def get_loss_sequence(self):
        return self.loss_sequence


class Mixup_LogisticRegression(BaseEstimator):
    def __init__(self, alpha):
        self.alpha = alpha
    
    def fit(self, X = None, y = None):
        num_of_features, num_of_labels = X.shape[1], y.max()+1
        X = torch.from_numpy(X.astype(np.float32))
        y = torch.from_numpy(y.astype(np.long))
        y = F.one_hot(y).float()
        my_dataset = TensorDataset(X, y) # create your datset
        my_dataloader = DataLoader(my_dataset, batch_size = 128) # create your dataloader

        model = Logistic_Reg_model(num_of_features, num_of_labels)
        criterion = nn.CrossEntropyLoss()
        
        optimizer = torch.optim.SGD(
                                    model.parameters(), 
                                    momentum=0.9, 
                                    lr=0.02, 
                                    weight_decay = 0.0001
                                    )
        
        loss_sequence = {}
        number_of_epochs = 100
        for epoch in range(number_of_epochs):
          for batch_ndx, sample in enumerate(my_dataloader):
            X, y = sample
            
            lam = get_lambda(alpha = self.alpha)
            X_mix, y_a, y_b = mixup_data(X, y, lam)
            y_hat = model(X_mix)
            loss = lam * criterion(y_hat, y_a) + (1 - lam) * criterion(y_hat, y_b)
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
        return (np.argmax(self.model(X).detach().numpy(), axis = 1) == y).mean()

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



class Adamixup_LogisticRegression(BaseEstimator):
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
        model = Logistic_Reg_model(num_of_features, num_of_labels)
        ad_generator = Adamix_lambda_generator(num_of_features)
        intrusion_classifier = Adamix_intrusion_classifier(num_of_features)

        criterion = nn.CrossEntropyLoss()
        
        optimizer = torch.optim.SGD(
                                    list(model.parameters()) + list(ad_generator.parameters()) + list(intrusion_classifier.parameters()), 
                                    momentum=0.9, 
                                    lr=0.02, 
                                    weight_decay = 0.0001
                                    )
        loss_sequence = {}
        number_of_epochs = 100
        for epoch in range(number_of_epochs):
          for batch_ndx, sample in enumerate(my_dataloader):
            X, y = sample
            random_index2 = torch.randperm(X.shape[0]) # shuffled index for X2, y2
            random_index3 = torch.randperm(X.shape[0]) # shuffled index for X3, y3
            #print(random_index2)
            #print(random_index3)
            #ipdb.set_trace()

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
        return (np.argmax(self.model(X).detach().numpy(), axis = 1) == y).mean()

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

class GenLabel_LogisticRegression(BaseEstimator):
    def __init__(self, alpha=1, gen_model="GM", gamma=1):
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
        m_array, s_array = estimate_gm(X_original, y_original) # numpy arrays
        
        # Train
        my_dataset = TensorDataset(X, y) # create your datset
        my_dataloader = DataLoader(my_dataset, batch_size = 128) # create your dataloader

        model = Logistic_Reg_model(num_of_features, num_of_labels)
        criterion = nn.CrossEntropyLoss()
        
        optimizer = torch.optim.SGD(
                                    model.parameters(), 
                                    momentum=0.9, 
                                    lr=0.02, 
                                    weight_decay = 0.0001
                                    )
        
        loss_sequence = {}
        number_of_epochs = 100
        for epoch in range(number_of_epochs):
          for batch_ndx, sample in enumerate(my_dataloader):
            X, y = sample
            
            lam = get_lambda(alpha = self.alpha)
            X_mix, y_a, y_b = mixup_data(X, y, lam) 
            y_hat = model(X_mix)
            # ipdb.set_trace()
            P_array = np.zeros((X_mix.shape[0], num_of_labels))
            for class_idx in range(num_of_labels):
              P_array[:, class_idx] = multivariate_normal(m_array[class_idx], 
                                                             s_array[class_idx]).pdf(X_mix)
            #ipdb.set_trace()                                                            
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
        return (np.argmax(self.model(X).detach().numpy(), axis = 1) == y).mean()

    def get_loss_sequence(self):
        return self.loss_sequence        

