import utils
import copy
import math
import faiss
import torch

import numpy as np

class upBinaryCluster(object):
  def __init__(self, dim, classNum):
    self.dim = dim
    self.classNum = classNum
    self.kmeans = faiss.Kmeans(self.dim, self.classNum, niter=0, min_points_per_centroid=1)

  def getPseudo(self, data, centroids, size, batch):
    posMatrix, negMatrix = np.zeros(batch), np.zeros(batch)
    self.kmeans.train(data, init_centroids=centroids)
    distance, indexs = self.kmeans.index.search(data,1)
    posD, negD = copy.deepcopy(distance), copy.deepcopy(distance)
    posD[indexs == 1] = np.inf
    negD[indexs == 0] = np.inf
    posIndex = np.argsort(posD.T)[0][:int(sum(posD < np.inf))]
    negIndex = np.argsort(negD.T)[0][:int(sum(negD < np.inf))]
    posMatrix[posIndex[:size]] = 1
    negMatrix[negIndex[:size]] = 1
    # distance = distance.T[0] / np.mean(distance.T[0])
    return posMatrix, negMatrix, None, None


class Estimator(object):
  def __init__(self, dim, classNum, maxEpochs):
    self.maxEpochs = maxEpochs
    self.binaryCluster = upBinaryCluster(dim, classNum)

  def getCluster(self, posData, anSig, ngSig, epoch, batch, clusterRate):
    # size = math.ceil(batch * (epoch/self.maxEpochs))
    size = math.ceil(batch * clusterRate * (epoch/self.maxEpochs)) if epoch <= self.maxEpochs else batch
    size = 1 if size < 1 else size

    posIdxs, negIdxs, dists, hardists =[], [], [], []
    
    for ans, neg in zip(anSig, ngSig):
      # posMtx, negMtx, dist, hardist = self.binaryCluster.getPseudo(copy.deepcopy(posData), np.array([ans,neg]), size, batch)
      posMtx, negMtx, dist, hardist = self.binaryCluster.getPseudo(copy.deepcopy(posData), utils.kmeans_plus(copy.deepcopy(anSig), 2, ans, random_state=42), size, batch)
      posIdxs.append(posMtx)
      negIdxs.append(negMtx)
      # dists.append(dist)
    posIdxs, negIdxs, dists = torch.Tensor(np.array(posIdxs)), torch.Tensor(np.array(negIdxs)), torch.Tensor(np.array(dists))
    posIdxs, negIdxs = torch.Tensor(np.array(posIdxs)), torch.Tensor(np.array(negIdxs))
    disIdxs = torch.where((posIdxs + negIdxs) > 0, torch.zeros_like(posIdxs), torch.ones_like(posIdxs))
    return posIdxs, negIdxs, torch.Tensor(disIdxs), None, None
