import torch
import copy
import numpy as np

from torch import nn
from torch.nn import ParameterList
from torch.nn.parameter import Parameter
from Networks.Trainer import EncoderGCN

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class MLP(nn.Module):
    def __init__(self, in_channels, num_classes=2):
        super(MLP, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(in_channels, num_classes),
            nn.Sigmoid()
        )

    def forward(self, input):
      return self.net(input)    

class Predictor(nn.Module):
    def __init__(self, num_features, num_hidden, layer_num, num_classes=2):
        super(Predictor, self).__init__()
        self.gnn = EncoderGCN(num_features, num_hidden, layer_num)
        self.mlp = MLP(num_hidden*2, num_classes)

    def forward(self, data):
        return self.mlp(self.gnn(data.x, data.edge_index, data.batch))


class Generator(nn.Module):
    def __init__(self, in_channels, num_hidden, graphs_num, gnnlayer_num, num_classes) -> None:
        super(Generator, self).__init__()

        self.where = MyWhere.apply
        self.TOPK = MyTopK.apply

        self.in_channels = in_channels
        self.predictor = Predictor(in_channels, num_hidden, gnnlayer_num, num_classes)

        self.perturbation_matrices = ParameterList([Parameter(torch.FloatTensor(in_channels, in_channels)) for _ in range(graphs_num)])
        for each in self.perturbation_matrices:
            torch.nn.init.xavier_uniform_(each.data, gain=5/3)

        self.perturbation_biases = ParameterList([Parameter(torch.FloatTensor(in_channels, in_channels)) for _ in range(graphs_num)])
        for each in self.perturbation_biases:
            torch.nn.init.xavier_uniform_(each.data, gain=5/3)

        self.masking_matrices = ParameterList([Parameter(torch.FloatTensor(in_channels, in_channels)) for _ in range(graphs_num)])
        for each in self.masking_matrices:
            torch.nn.init.xavier_uniform_(each.data, gain=5/3)
    
    def getPrediction(self, orignal, augmented):
        return self.predictor(orignal), self.predictor(augmented)
    
    def forward(self, graphs):
        perturbation_matrices = tuple([self.perturbation_matrices[id] for id in graphs.id])
        perturbation_matrices = torch.block_diag(*perturbation_matrices)

        perturbation_biases = tuple([self.perturbation_biases[id] for id in graphs.id])
        perturbation_biases = torch.block_diag(*perturbation_biases)

        masking_matrices = [self.masking_matrices[int(id)] for id in graphs.id]
        masking_matrices = torch.cat(masking_matrices, dim=0)

        graphs_augmentation = copy.deepcopy(graphs)
        
        # edge perturbation
        values = torch.Tensor([1 for _ in range(graphs.edge_index.size()[1])])
        adjs = torch.sparse_coo_tensor(graphs.edge_index, values.to(device), (graphs.num_nodes, graphs.num_nodes), dtype=torch.float)
        adjs_dense = adjs.to_dense()
        perturbation_adjs = torch.mm(perturbation_matrices, adjs_dense)+perturbation_biases
        perturbation_adjs = torch.sigmoid(perturbation_adjs)
        perturbation_adjs = self.where(perturbation_adjs)
        perturbation_adjs_sparse = perturbation_adjs.to_sparse()
        graphs_augmentation.edge_index = perturbation_adjs_sparse.indices()

        # feature mask
        masking_matrices = torch.sigmoid(masking_matrices)
        masking_matrices = self.where(masking_matrices)
        masked_feature = torch.mul(masking_matrices, graphs.x)
        graphs_augmentation.x = masked_feature

        # new version EDGE
        # values = torch.Tensor([1 for _ in range(graphs.edge_index.size()[1])])
        # adjs = torch.sparse_coo_tensor(graphs.edge_index, values.to(device), (graphs.num_nodes, graphs.num_nodes), dtype=torch.float)
        # adjs_dense = adjs.to_dense()
        # perturbation_adjs = torch.mm(perturbation_matrices, adjs_dense)+perturbation_biases
        # perturbation_adjs = torch.sigmoid(perturbation_adjs)
        # perturbation_mask = tuple([torch.ones_like(self.perturbation_matrices[id]) for id in graphs.id])
        # perturbation_mask = torch.block_diag(*perturbation_mask).fill_diagonal_(0)
        # perturbation_adjs = torch.mul(perturbation_adjs, perturbation_mask)
        # perturbation_adjs, perturbation_feature = self.TOPK(perturbation_adjs)
        # perturbation_adjs,_ = self.TOPK(perturbation_adjs)
        # perturbation_adjs_sparse = perturbation_adjs.to_sparse()
        # graphs_augmentation.edge_index = perturbation_adjs_sparse.indices()

        # # new bersion FEATURE
        # perturbation_feat = [perturbation_feature[index:index+self.in_channels, index:index+self.in_channels] for index in range(0, len(perturbation_feature), self.in_channels)]
        # perturbation_feat = torch.cat(perturbation_feat, dim=0)
        # graphs_augmentation.x = perturbation_feat

        predicted_results, augmentation_predicted_results = self.getPrediction(graphs, graphs_augmentation)

        return adjs_dense, perturbation_adjs, masking_matrices, predicted_results, augmentation_predicted_results, graphs_augmentation


# Customer KNN Function
class MyTopK(torch.autograd.Function):
    @staticmethod
    def forward(inGraphs, k=5):
        adjacency = torch.zeros_like(inGraphs)
        feature = torch.zeros_like(inGraphs)
        topk_idx = torch.topk(inGraphs, k=k)
        for index, ele in enumerate(topk_idx.indices):
            feature[index, ele] = topk_idx.values[index]
        feature = feature + feature.T
        adjacency = torch.where(feature>0, torch.ones_like(feature), torch.zeros_like(feature))
        return adjacency, feature
    
    @staticmethod
    def setup_context(ctx, inputs, output):
        adj, feat = output
        inputs, = inputs
        ctx.save_for_backward(inputs, adj, feat)

    @staticmethod
    def backward(ctx, grad_adj, grad_feat):
        input, adj, feat = ctx.saved_tensors
        # grad_input = torch.mul(adj, torch.ones_like(grad_feat))
        return grad_adj

# Customer where Function
class MyWhere(torch.autograd.Function):
    @staticmethod
    def forward(x):
        result = torch.where(x<=0.5, torch.zeros_like(x), torch.ones_like(x))
        return result
    
    @staticmethod
    def setup_context(ctx, inputs, output):
        ctx.save_for_backward(output)

    @staticmethod
    def backward(ctx, output):
        # grad_input, = ctx.saved_tensors
        return output

