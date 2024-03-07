from Networks import LinearNetwork
import utils
import torch
import dataloader
import numpy as np

from sklearn.metrics import f1_score, accuracy_score, roc_auc_score
from sklearn.model_selection import StratifiedKFold
from torch.optim import Adam
from torch.utils.data import DataLoader


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def mlp_evaluator(x, y, learning_rate=5e-3):

    x, y = torch.Tensor(x), torch.Tensor(y)
        
    
    results = [[],[],[]]
    skf = StratifiedKFold(n_splits=5, shuffle=True)
    fold = 1
    for train_index, test_index in skf.split(x, y):
        model = LinearNetwork.MLP(x.shape[1]).to(device)
        loss_function = torch.nn.CrossEntropyLoss()
        optimizer = Adam(model.parameters(), lr=learning_rate, weight_decay=1e-5)
        data_loader = DataLoader(dataloader.MLPDataset(x[train_index], y[train_index]), batch_size=32, shuffle=True)
        data_loader_eval = DataLoader(dataloader.MLPDataset(x[test_index], y[test_index]), batch_size=32, shuffle=True)
        temp_result = []

        for epoch in range(0, 100):
            utils.adjust_learning_rate(optimizer, epoch, learning_rate)
            train(model, optimizer, loss_function, data_loader)
            
        acc, f1, auc = test(model, data_loader_eval)
        results[0].append(acc)
        results[1].append(f1)
        results[2].append(auc)
        # print(f"Fold-{fold} --> Acc: {acc}, F1Ma: {f1}, AUC: {auc}")
        fold += 1
    return np.mean(results[0]), np.std(results[0]), np.mean(results[1]), np.std(results[1]), np.mean(results[2]), np.std(results[2])

def train(model, optimizer, loss_function, data_loader):

    model.train()

    for data in data_loader:  # Iterate in batches over the training dataset.
        out = model(data[0].to(device))
        loss = loss_function(out, data[1].type(torch.LongTensor).to(device)) 
        loss.backward()  
        optimizer.step()  
        optimizer.zero_grad()
    return loss.item()

def test(model, data_loader_eval):
    model.eval()
    preds = []
    labels = []
    for data in data_loader_eval:  
        out = model(data[0].to(device))
        pred = out.argmax(dim=1)
        preds.extend(pred.detach().cpu())
        labels.extend(data[1])
    return accuracy_score(labels,preds), \
           f1_score(labels, preds, average = 'macro'), \
           roc_auc_score(labels, preds)