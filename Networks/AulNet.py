import Networks.LinearNetwork as Classifier
import torch
import dataloader
import torch.nn as nn
import utils
import numpy as np
import copy

from tqdm import tqdm

from torch.utils.data import DataLoader
from torch.nn import functional as F

device = torch.device("cuda" if torch.cuda.is_available() else 'cpu')


class AUEstimator:
    def __init__(self, in_channels, learning_rate, weight_decay, wtrain_epoch, reward, warmup, k=2):
        super(AUEstimator, self).__init__()
        self.learning_rate = learning_rate
        self.model = Classifier.WeightMLP(in_channels*2, num_classes=k+1).to(device)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=learning_rate, weight_decay=weight_decay)
        self.loss_function = nn.CrossEntropyLoss()
        self.wtrain_epoch = wtrain_epoch
        self.reward = reward
        self.k = k
        self.data = None
        self.y = None
        self.warmup = warmup
        self.flag = False
    
    def weightCal(self, anchors, negSample, epoch):
        uncertainties = []
        for anchor in anchors:
            pLabels = torch.Tensor(utils.binaryPartition(anchor, negSample, self.k).labels_).type(torch.LongTensor).to(device)
            uncertainties.append(self.uncertaintyEstimator(torch.Tensor(mergeDataset(anchor, negSample)).to(device), pLabels, epoch))
        return torch.Tensor(np.array(uncertainties)).to(device)
    

    def uncertaintyEstimator(self, negSample, pLabels, epoch):
        if epoch <= self.warmup:
            if self.data == None and self.y == None:
                self.data, self.y = copy.deepcopy(negSample), copy.deepcopy(pLabels)
            else:
                self.data, self.y = torch.cat((self.data, negSample), dim=0), torch.cat((self.y, pLabels), dim=0)
                          
            return [1 for _ in range(negSample.shape[0])]
        
        else:
            if not self.flag:
                trainset = DataLoader(dataloader.MLPDataset(self.data.to(device), self.y.to(device)), batch_size=32, shuffle=True)
                for times in range(20):
                    train(self.model, self.optimizer, self.loss_function, trainset, times, self.reward)
                self.flag = True
            data_loader = DataLoader(dataloader.MLPDataset(negSample.to(device), pLabels.to(device)), batch_size=32, shuffle=True)
            return test(self.model, data_loader)


def mergeDataset(anchor: np.ndarray, negSamples: np.ndarray):
    return np.concatenate((np.tile(anchor, (negSamples.shape[0], 1)), negSamples), axis=1)

def train(model, optimizer, loss_function, data_loader, epoch, reward=5):

    model.train()

    for data in data_loader:
        outputs = model(data[0])
        
        if epoch >= 5:
            outputs = F.softmax(outputs, dim=1)
            outputs, reservation = outputs[:,:-1], outputs[:,-1]
            gain = torch.gather(outputs, dim=1, index=data[1].unsqueeze(1)).squeeze()
            doubling_rate = (gain.add(reservation.div(reward))).log()

            loss = -doubling_rate.mean()
        else:
            loss = loss_function(outputs[:,:-1], data[1])
        
        loss.backward()  
        optimizer.step()  
        optimizer.zero_grad()

def test(model, data_loader_eval):
    abortion_results = []
    model.eval()
    for data in data_loader_eval:  
        outputs = model(data[0])
        outputs = F.softmax(outputs, dim=1)
        outputs, reservation = outputs[:,:-1], outputs[:,-1]
        abortion_results.extend(list(reservation.detach().cpu().numpy()))
    return np.array(abortion_results / np.mean(abortion_results))
