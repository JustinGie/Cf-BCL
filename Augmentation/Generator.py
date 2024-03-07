import utils
import tqdm
import torch

from Networks.AugNet import Generator

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class AugGen(object):
    def __init__(self, in_channels, hidden_num, graphs_num, layers, classes):
        self.Model = Generator(in_channels, hidden_num, graphs_num, gnnlayer_num=layers, num_classes=classes).to(device)
        self.Optimizer = torch.optim.Adam(
            params=self.Model.parameters(),
            lr=5e-3,
            weight_decay=1e-5
        )
    
    def train(self, dataloader, epochs):
        self.Model.train()
        pbar = tqdm.tqdm(range(1, epochs+1))
        for epoch in pbar:
            pbar.set_description('Epoch %d...' % epoch)
            # utils.adjust_learning_rate(self.Optimizer, epoch, 5e-3)
            for batch in dataloader:
                batch.to(device)
                self.Optimizer.zero_grad()
                adjs_dense, perturbation_adjs, perturbation_feat, predicted_results, perturbation_predicted_results, _ = self.Model(batch)
                sim_loss = self.similarity_loss(perturbation_adjs-adjs_dense, perturbation_feat)
                kl_loss = self.kl_div(predicted_results, perturbation_predicted_results)
                loss = sim_loss-kl_loss
                loss.backward()
                self.Optimizer.step()
                pbar.set_postfix(sim_loss=sim_loss.item(), kl_loss=kl_loss.item())
            # print(batch.edge_index.shape)
            # print(new.edge_index.shape)
   
    def getGraph(self, dataset):
        with torch.no_grad():
            self.Model.eval()
            _, _, _, _, _, augGraphs = self.Model(dataset)
        return augGraphs
    
    def getGraphCasCade(self, dataset):
        self.Model.train()
        adjs_dense, perturbation_adjs, perturbation_feat, predicted_results, perturbation_predicted_results, augGraphs = self.Model(dataset)
        sim_loss = self.similarity_loss(perturbation_adjs-adjs_dense, perturbation_feat)
        kl_loss = self.kl_div(predicted_results, perturbation_predicted_results)
        return sim_loss, kl_loss, augGraphs
                
    def similarity_loss(self, diff_edge, diff_feat):
        raise NotImplementedError

    def kl_div(self, predicted_results, perturbation_predicted_results):
        raise NotImplementedError


class NegGen(AugGen):
    def similarity_loss(self, diff_edge, diff_feat):
        edgeNorm = torch.linalg.matrix_norm(diff_edge, ord='fro') / torch.linalg.matrix_norm(torch.ones_like(diff_feat), ord='fro')
        featNorm = torch.linalg.matrix_norm(diff_feat, ord='fro') / torch.linalg.matrix_norm(torch.ones_like(diff_feat), ord='fro')
        return edgeNorm + featNorm

    def kl_div(self, predicted_results, perturbation_predicted_results):
        predicted_results = predicted_results.softmax(dim=1)
        perturbation_predicted_results = perturbation_predicted_results.log_softmax(dim=1)  
        loss_func = torch.nn.KLDivLoss(reduction='batchmean')
        kl_loss = loss_func(perturbation_predicted_results, predicted_results)
        return kl_loss


class PosGen(AugGen):
    def similarity_loss(self, diff_edge, diff_feat):
        edgeNorm = torch.linalg.matrix_norm(diff_edge, ord='fro') / torch.linalg.matrix_norm(torch.ones_like(diff_feat), ord='fro')
        featNorm = torch.linalg.matrix_norm(diff_feat, ord='fro') / torch.linalg.matrix_norm(torch.ones_like(diff_feat), ord='fro')
        return edgeNorm + featNorm

    def kl_div(self, predicted_results, perturbation_predicted_results):
        predicted_results = predicted_results.softmax(dim=1)
        perturbation_predicted_results = perturbation_predicted_results.log_softmax(dim=1)  
        loss_func = torch.nn.KLDivLoss(reduction='batchmean')
        kl_loss = loss_func(perturbation_predicted_results, predicted_results)
        return -kl_loss
