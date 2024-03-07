import utils
import torch
import faiss
import Networks.AulNet as AUL
import torch.nn.functional as F
from torch_geometric.nn import GCNConv,global_add_pool, global_mean_pool
from InCluster.InClustering import upBinaryCluster


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


class EncoderGCN(torch.nn.Module):
    def __init__(self, num_features, dim, num_gc_layers):
        super(EncoderGCN, self).__init__()
        self.num_gc_layers = num_gc_layers
        self.convs = torch.nn.ModuleList()
        self.bns = torch.nn.ModuleList()

        for i in range(num_gc_layers):
            if i == 0:
                nn = GCNConv(num_features, dim)
            else:
                nn = GCNConv(dim, dim)
            self.convs.append(nn)
            bn = torch.nn.BatchNorm1d(dim)
            self.bns.append(bn)
          
    def embedding(self, x, edge_index, batch):
        if x is None:
            x = torch.ones((batch.shape[0], 1)).to(edge_index.device)
        xs = []
        for i in range(self.num_gc_layers):
            x = F.relu(self.convs[i](x, edge_index))
            x = self.bns[i](x)
            xs.append(x)
        return  xs
    
    def forward(self, x, edge_index, batch):
        xs = self.embedding(x, edge_index, batch)
        xpool = [global_add_pool(x, batch) for x in xs]
        x = torch.cat(xpool, 1)
        return x

class Model(torch.nn.Module):
    def __init__(self, encoder: EncoderGCN, num_hidden, num_proj_hidden, num_layer, tau: float = 0.5):
        super(Model, self).__init__()
        self.encoder = encoder
        self.tau = tau

        self.fc1 = torch.nn.Linear(num_hidden*num_layer, num_proj_hidden)
        self.fc2 = torch.nn.Linear(num_proj_hidden, num_hidden)
        self.cluster = upBinaryCluster(num_hidden*2, num_layer)

    def forward(self, x, edge_index, batch) -> torch.Tensor:
        return self.encoder(x, edge_index, batch)

    def projection(self, z: torch.Tensor) -> torch.Tensor:
        z = F.elu(self.fc1(z))
        return self.fc2(z)
    
    def pos_semi_loss(self, h1: torch.Tensor, h2: torch.Tensor, est, epoch):
        uncertainty = est.weightCal(h1.detach().cpu().numpy(), h2.detach().cpu().numpy(), epoch)
        weights = (1/torch.mean(uncertainty)) * uncertainty
        x_abs = h1.norm(dim=1)
        x_aug_abs = h2.norm(dim=1)

        sim_matrix = torch.einsum('ik,jk->ij', h1, h2) / torch.einsum('i,j->ij', x_abs, x_aug_abs)
        sim_matrix = torch.exp(sim_matrix / self.tau)

        second_t = weights.to(device)*sim_matrix
        loss = sim_matrix.diag() / (sim_matrix.diag() + (second_t.sum(dim=1) - second_t.diag()))
        return loss
    
    def neg_semi_loss(self, h1: torch.Tensor, h2: torch.Tensor, h3:torch.Tensor):
        pos1_abs = h1.norm(dim=1)
        pos2_abs = h2.norm(dim=1)
        neg_abs = h3.norm(dim=1)

        pos_sim_matrix = torch.einsum('ik,jk->ij', h1, h2) / torch.einsum('i,j->ij', pos1_abs, pos2_abs)
        pos_sim_matrix = torch.exp(pos_sim_matrix / self.tau)

        neg_sim_matrix = torch.einsum('ik,jk->ij', h1, h3) / torch.einsum('i,j->ij', pos1_abs, neg_abs)
        neg_sim_matrix = torch.exp(neg_sim_matrix / self.tau)

        loss = pos_sim_matrix.diag() / (pos_sim_matrix.diag() + neg_sim_matrix.diag())
        return loss


    def loss(self, pG1: torch.Tensor, pG2: torch.Tensor, nG: torch.Tensor, anSig: torch.Tensor):
        
        self.getPair(anSig)
        h1 = self.projection(pG1)
        h2 = self.projection(pG2)
        h3 = self.projection(nG)

        # posLoss = (self.pos_semi_loss(h1, h2, estimator, epoch) + self.pos_semi_loss(h2, h1, estimator, epoch)) * 0.5
        negLoss = (self.neg_semi_loss(h1, h2, h3) + self.neg_semi_loss(h2, h1, h3)) * 0.5
        loss = - torch.log(negLoss).mean()
        # loss = - torch.log(posLoss).mean() - torch.log(negLoss).mean()
        return loss

    
    def get_emb(self, x: torch.Tensor,
                edge_index: torch.Tensor, batch):
        orginal_emb = self.encoder.embedding(x, edge_index, batch)
        xmean = [x.sum(1) for x in orginal_emb] 
        xs = xmean[0] 
        for i in range(1,len(orginal_emb)):
            xs = xs + xmean[i]
        
        return xs
    
    def emb_avg(self, graph_emb, num_layer, batch, k):

        graph_emb = graph_emb.detach().cpu().numpy()
        index = faiss.IndexFlatL2(graph_emb.shape[1])
        index.add(graph_emb) 
        _, I = index.search(graph_emb, 5)
        graph_emb_new = 0
        for i in range(1, k):
            graph_emb_new = graph_emb_new + graph_emb[I[:,i],:] 
        graph_emb_new = torch.tensor(graph_emb_new).to(batch.device)
        node_emb  = graph_emb_new[batch] 
        node_emb  = F.normalize(node_emb,dim=1)
        return node_emb
        
    
    def get_emb_avg(self, x, edge_index, batch, k):
        
        orginal_emb = self.encoder.embedding(x, edge_index, batch)
        num_layer = len(orginal_emb)
        
        graph_emb = global_mean_pool(orginal_emb[0], batch)
        for i in range(1,len(orginal_emb)):
            graph_emb = graph_emb + global_mean_pool(orginal_emb[i], batch)

        xs = orginal_emb[0]
        for i in range(1,len(orginal_emb)):
            xs = xs + orginal_emb[i]
            
        xgraph = self.emb_avg(graph_emb, num_layer, batch, k)
        xs = xs * xgraph
        
        xs_mean = xs.sum(1)
        return xs_mean
    

