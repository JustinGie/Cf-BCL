import math
import utils
import torch

import numpy as np
import torch.nn.functional as F

from InCluster.InClustering import Estimator
from torch_geometric.nn import GCNConv, TopKPooling, Sequential

from Networks.AugNet import Generator

device = torch.device("cuda" if torch.cuda.is_available() else 'cpu')

class AttentionReadout(torch.nn.Module):
    def __init__(self, output_dim, hidden_dim, dropout=0.1):
        super().__init__()
        self.embed_query = torch.nn.Linear(hidden_dim, output_dim)
        self.embed_key = torch.nn.Linear(hidden_dim, output_dim)
        self.dropout = torch.nn.Dropout(dropout)
    
    def forward(self, x, batch, node_axis=-2):
        x = x.reshape(batch[-1].item()+1, -1, x.shape[-1])
        x_q = self.embed_query(x.mean(node_axis))
        x_k = self.embed_key(x)
        x_graphattention = torch.sigmoid(torch.matmul(x_q.unsqueeze(1), x_k.transpose(2,1)) / np.sqrt(x_q.shape[-1])).squeeze(2)
        return (x * self.dropout(x_graphattention.transpose(2,1))).mean(node_axis), x_graphattention


class EncoderGCN(torch.nn.Module):
    def __init__(self, num_features, dim, num_gc_layers, ratio):
        super(EncoderGCN, self).__init__()
        self.num_gc_layers = num_gc_layers
        self.convs = torch.nn.ModuleList()
        self.attentionReadout = torch.nn.ModuleList()

        for i in range(num_gc_layers):
            if i == 0:
                nn = GCNConv(num_features, dim)
            else:
                nn = GCNConv(dim, dim)
            self.convs.append(nn)
            self.attentionReadout.append(AttentionReadout(dim, dim, dropout=0.1))


    def embedding(self, x, edge_index, batch):
        if x is None:
            x = torch.ones((batch.shape[0], 1)).to(edge_index.device)
        scores = []
        readouts = []
        for i in range(self.num_gc_layers):
            x = self.convs[i](x, edge_index)
            readout, _ = self.attentionReadout[i](x, batch)
            readouts.append(readout)
            # scores.append(score)
        return  torch.concat(readouts, dim=1), scores
    
    
    def forward(self, x, edge_index, batch):
        readout, _ = self.embedding(x, edge_index, batch)
        return readout
    

class Model(torch.nn.Module):
    def __init__(self, encoder: EncoderGCN, posGen: Generator, negGen: Generator, num_hidden, num_proj_hidden, num_layer, batch, maxEpochs, tau: float = 0.5, estRatio: float = 0.5):
        super(Model, self).__init__()
        self.encoder = encoder
        self.tau = tau

        self.fc1 = torch.nn.Linear(num_hidden*num_layer, num_proj_hidden)
        self.fc2 = torch.nn.Linear(num_proj_hidden, num_hidden)

        self.estor = Estimator(num_hidden, num_layer, maxEpochs*estRatio)

        self.posGen = posGen
        self.negGen = negGen

    def forward(self, data):
        posSimLoss, posKLoss, posGraph = getAugGraph(self.posGen, data, 'POS')
        negSimLoss, negKLoss, negGraph = getAugGraph(self.negGen, data, 'NEG')
        return self.encoder(data.x, data.edge_index, data.batch), \
            posSimLoss, posKLoss, self.encoder(posGraph.x, posGraph.edge_index, posGraph.batch), \
                negSimLoss, negKLoss, self.encoder(negGraph.x, negGraph.edge_index, negGraph.batch)

    def projection(self, z: torch.Tensor) -> torch.Tensor:
        z = F.elu(self.fc1(z))
        return self.fc2(z)
    
    def semi_loss(self, h1: torch.Tensor, h2: torch.Tensor, h3:torch.Tensor, epoch, clusterRate):
        posMx, negMx, disMx, distMx, hardistMx = self.getPair(h1.detach().cpu().numpy(), h2.detach().cpu().numpy(), h3.detach().cpu().numpy(), epoch, clusterRate)
        an_abs = h1.norm(dim=1)
        pos_abs = h2.norm(dim=1)
        neg_abs = h3.norm(dim=1)

        pos_sim_matrix = torch.einsum('ik,jk->ij', h1, h2) / torch.einsum('i,j->ij', an_abs, pos_abs)
        pos_sim_matrix = torch.exp(pos_sim_matrix / self.tau)

        neg_sim_matrix = torch.einsum('ik,jk->ij', h1, h3) / torch.einsum('i,j->ij', an_abs, neg_abs)
        neg_sim_matrix = torch.exp(neg_sim_matrix / self.tau)

        nt_matrix = posMx * pos_sim_matrix

        loss = nt_matrix.sum(dim=1) / (pos_sim_matrix.sum(dim=1) + neg_sim_matrix.diag())
        return loss
    
    def getPair(self, posData: torch.Tensor, anSig: torch.Tensor, ngSig: torch.Tensor, epoch, clusterRate):
        posIdxs, negIdxs, disIdxs, _, _ = self.estor.getCluster(posData, anSig, ngSig, epoch, anSig.shape[0], clusterRate)
        posIdxs, negIdxs, disIdxs = posIdxs.fill_diagonal_(1), negIdxs.fill_diagonal_(0), disIdxs.fill_diagonal_(0)
        return posIdxs.to(device), negIdxs.to(device), disIdxs.to(device), None, None

    def loss(self, pG: torch.Tensor, nG: torch.Tensor, anSig: torch.Tensor, epoch, clusterRate):
        aEmbeding = self.projection(anSig) # anchor
        pEmbeding = self.projection(pG) # positve aug
        nEmbeding = self.projection(nG) # negative aug
        
        negLoss = self.semi_loss(pEmbeding, aEmbeding, nEmbeding, epoch, clusterRate)
        loss = - torch.log(negLoss).mean()
        return loss


def getAugGraph(model, data, flag='POS'):
        adjs_dense, perturbation_adjs, perturbation_feat, predicted_results, perturbation_predicted_results, augGraph = model(data)
        sim_loss = similarity_loss(perturbation_adjs-adjs_dense, perturbation_feat)
        kl_loss = kl_div(predicted_results, perturbation_predicted_results)

        if flag == 'POS':
            return sim_loss, -kl_loss, augGraph
        else:
            return sim_loss, kl_loss, augGraph
 
def similarity_loss(diff_edge, diff_feat):
    edgeNorm = torch.linalg.matrix_norm(diff_edge, ord='fro') / torch.linalg.matrix_norm(torch.ones_like(diff_feat), ord='fro')
    featNorm = torch.linalg.matrix_norm(diff_feat, ord='fro') / torch.linalg.matrix_norm(torch.ones_like(diff_feat), ord='fro')
    return edgeNorm + featNorm

def kl_div(predicted_results, perturbation_predicted_results):
    predicted_results = predicted_results.softmax(dim=1)
    perturbation_predicted_results = perturbation_predicted_results.log_softmax(dim=1)  
    loss_func = torch.nn.KLDivLoss(reduction='batchmean')
    kl_loss = loss_func(perturbation_predicted_results, predicted_results)
    return kl_loss
