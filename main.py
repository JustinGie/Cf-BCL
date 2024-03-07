import copy
import torch
import utils
import argparse
import dataloader

import numpy as np

from tqdm import tqdm
from evaluation import mlp_evaluator
from torch_geometric.loader import DataLoader
from Augmentation.Generator import NegGen, PosGen
from Networks.ContrastiveL import EncoderGCN, Model
from Networks.AulNet import AUEstimator
from Networks.AugNet import Generator

device = torch.device("cuda" if torch.cuda.is_available() else 'cpu')

def train(model, optimizer, trainLoader, epoch, clusterRate):
    model.train()
    Loss, negSL, negKL, posSL, posKL = 0,0,0,0,0
    for data in trainLoader:
        data = data.to(device)
        optimizer.zero_grad()

        anSig, psl, pkl, pG, nsl, nkl, nG = model(data)

        contrastiveLoss = model.loss(pG, nG, anSig, epoch, clusterRate)
        loss = contrastiveLoss + (psl - pkl) + (nsl - nkl)
        loss.backward()
        optimizer.step()
        Loss += loss.item()
        negSL += nsl.item()
        negKL += nkl.item()
        posSL += psl.item()
        posKL += pkl.item()
    return Loss / len(trainLoader), negSL / len(trainLoader), negKL / len(trainLoader), posSL / len(trainLoader), posKL / len(trainLoader)

def eval(model, evalLoader):
    model.eval()
    embeds = []
    labels = []
    
    for data in evalLoader:
        data = data.to(device)
        embeddings, _, _, _, _, _, _ = model(data)
        labels.append(data.y.detach().cpu().numpy())
        embeds.append(embeddings.detach().cpu().numpy())
    embeds = np.concatenate(embeds, 0)
    labels = np.concatenate(labels, 0)

    return mlp_evaluator(embeds, labels)

def trainAndEval(trainLoader, evalLoader, augFirst, augThird, args):
    encoder = EncoderGCN(args.ROI, args.hiddenNum, args.layerNum, args.ratio)
    posGen = Generator(augFirst, args.hiddenNum, augThird, 2, 2)
    negGen = Generator(augFirst, args.hiddenNum, augThird, 2, 2)
    model = Model(encoder, posGen, negGen, args.hiddenNum, args.hiddenMIDNum, args.layerNum, args.batchSize, args.Epochs, args.tau, args.estRatio).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learningRate, weight_decay=args.weightDecay)

    # Training
    pbar = tqdm(range(args.Epochs))
    bestLoss = 1e9
    checkPoint = model.state_dict()
    for epoch in pbar:
        pbar.set_description('Epoch %d...' % epoch)
        utils.adjust_learning_rate(optimizer, epoch, args.learningRate)
        loss, nsl, nkl, psl, pkl = train(model, optimizer, trainLoader, epoch, args.clusterRate)
        pbar.set_postfix(Loss=loss)
    
        if loss < bestLoss:
            checkPoint = model.state_dict()
            bestLoss = loss

    # Evaluating
    print(f"=== Result===")
    # acc, acc_std, f1ma, f1ma_std, auc, auc_std = eval(model, evalLoader, args.testLr)
    model.load_state_dict(checkPoint)
    acc, acc_std, f1ma, f1ma_std, auc, auc_std = eval(model, evalLoader)
    print( f'ACC: {acc:.4f}±{acc_std:.4f}, F1Ma: {f1ma:.4f}±{f1ma_std:.4f}, AUC: {auc:.4f}±{auc_std:.4f}')

    return None

    


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', type=str, default='data')
    parser.add_argument('--dataset', type=str, default='ADHD')
    parser.add_argument('--modality', type=str, default='fmri')

    parser.add_argument('--batchSize', type=int, default=32)
    parser.add_argument('--Epochs', type=int, default=100)
    parser.add_argument('--ROI', type=int, default=116)
    parser.add_argument('--hiddenNum', type=int, default=128)
    parser.add_argument('--layerNum', type=int, default=2)
    parser.add_argument('--hiddenMIDNum', type=int, default=128)
    parser.add_argument('--ratio', type=float, default=0.6)
    parser.add_argument('--tau', type=float, default=0.5)
    parser.add_argument('--estRatio', type=float, default=0.2)
    parser.add_argument('--clusterWarmup', type=float, default=0.05)
    parser.add_argument('--clusterRate', type=float, default=0.3)
    parser.add_argument('--learningRate', type=float, default=2e-3)
    parser.add_argument('--weightDecay', type=float, default=2e-5)
    parser.add_argument('--runtimes', type=int, default=1)
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()

    utils.setup_seed(args.seed)

    dataset = dataloader.MyOwnDataset(args.root, args.dataset, args.modality)
    augloader = DataLoader(dataset, batch_size=args.batchSize, shuffle=True)

    # Generator 2-Positive and 1-Negative
    negGene = NegGen(dataset[0].x.shape[1], args.hiddenNum, len(dataset), 2, 2)
    posGene = PosGen(dataset[0].x.shape[1], args.hiddenNum, len(dataset), 2, 2)


    # Contrastive Learning
    for i in range(args.runtimes):

        dataset.shuffle()
        trainLoader = DataLoader(dataset, batch_size=args.batchSize, shuffle=True)
        evalLoader = DataLoader(dataset, batch_size=args.batchSize, shuffle=True)

        trainAndEval(trainLoader, evalLoader, dataset[0].x.shape[1], len(dataset), args)