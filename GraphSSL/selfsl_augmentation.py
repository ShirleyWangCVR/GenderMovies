import torch.nn as nn
import scipy.sparse as sp
import torch.nn.functional as F
import numpy as np
import torch

from torch_geometric.data import Data, Dataset, Batch
from torch_geometric.nn import GATConv
from torch_geometric.loader import DataLoader
from torch_geometric.utils import from_networkx, to_networkx, to_dense_adj, dense_to_sparse, subgraph



### Graph Augmentations ###

class UniformSample():
    """
    Uniformly node dropping on the given graph or batched graphs. 
    Class objects callable via method :meth:`views_fn`.
    
    Args:
        ratio (float, optinal): Ratio of nodes to be dropped. (default: :obj:`0.1`)
    """

    def __init__(self, ratio=0.1):
        self.ratio = ratio
    
    def do_trans(self, data):
        
        node_num, _ = data.x.size()
        _, edge_num = data.edge_index.size()
        
        keep_num = int(node_num * (1-self.ratio))
        idx_nondrop = torch.randperm(node_num)[:keep_num]
        mask_nondrop = torch.zeros_like(data.x[:,0]).scatter_(0, idx_nondrop, 1.0).bool()
        
        edge_index, edge_attr = subgraph(mask_nondrop, data.edge_index, edge_attr=data.edge_attr, 
                                         relabel_nodes=True, num_nodes=node_num)

        names = [data.name[i] for i in range(len(mask_nondrop)) if mask_nondrop[i]]
        return Data(x=data.x[mask_nondrop], edge_index=edge_index, name=names, edge_attr=edge_attr)
    
    def views_fn(self, data):
        """
        Method to be called when :class:`EdgePerturbation` object is called.
        
        Args:
            data (:class:`torch_geometric.data.Data`): The input graph or batched graphs.
            
        :rtype: :class:`torch_geometric.data.Data`.  
        """

        if isinstance(data, Batch):
            dlist = [self.do_trans(d) for d in data.to_data_list()]
            return Batch.from_data_list(dlist)
        elif isinstance(data, Data):
            return self.do_trans(data)
    
class RWSample():
    """
    Subgraph sampling based on random walk on the given graph or batched graphs.
    Class objects callable via method :meth:`views_fn`.
    
    Args:
        ratio (float, optional): Percentage of nodes to sample from the graph.
            (default: :obj:`0.1`)
        add_self_loop (bool, optional): Set True to add self-loop to edge_index.
            (default: :obj:`False`)
    """

    def __init__(self, ratio=0.8, add_self_loop=False):
        self.ratio = ratio
        self.add_self_loop = add_self_loop
    
    def do_trans(self, data):
        node_num, _ = data.x.size()
        sub_num = int(node_num * self.ratio)

        if self.add_self_loop:
            sl = torch.tensor([[n, n] for n in range(node_num)]).t()
            edge_index = torch.cat((data.edge_index, sl), dim=1)
        else:
            edge_index = data.edge_index.detach().clone()

        idx_sub = [np.random.randint(node_num, size=1)[0]]
        idx_neigh = set([n.item() for n in edge_index[1][edge_index[0]==idx_sub[0]]])

        count = 0
        while len(idx_sub) <= sub_num:
            count = count + 1
            if count > node_num:
                break
            if len(idx_neigh) == 0:
                break
            sample_node = np.random.choice(list(idx_neigh))
            if sample_node in idx_sub:
                continue
            idx_sub.append(sample_node)
            idx_neigh.union(set([n.item() for n in edge_index[1][edge_index[0]==idx_sub[-1]]]))

        idx_sub = torch.LongTensor(idx_sub).to(data.x.device)
        mask_nondrop = torch.zeros_like(data.x[:,0]).scatter_(0, idx_sub, 1.0).bool()        
        edge_index, edge_attr = subgraph(mask_nondrop, data.edge_index, edge_attr=data.edge_attr, 
                                         relabel_nodes=True, num_nodes=node_num)
        
        names = [data.name[i] for i in range(len(mask_nondrop)) if mask_nondrop[i]]
        return Data(x=data.x[mask_nondrop], edge_index=edge_index, name=names, edge_attr=edge_attr)
    
    def views_fn(self, data):
        """
        Method to be called when :class:`EdgePerturbation` object is called.
        
        Args:
            data (:class:`torch_geometric.data.Data`): The input graph or batched graphs.
            
        :rtype: :class:`torch_geometric.data.Data`.  
        """

        if isinstance(data, Batch):
            dlist = [self.do_trans(d) for d in data.to_data_list()]
            return Batch.from_data_list(dlist)
        elif isinstance(data, Data):
            return self.do_trans(data)
    
class NodeAttrMask():
    """
    Node attribute masking on the given graph or batched graphs. 
    Class objects callable via method :meth:`views_fn`.
    
    Args:
        mode (string, optinal): Masking mode with three options:
            :obj:`"whole"`: mask all feature dimensions of the selected node with a Gaussian distribution;
            :obj:`"partial"`: mask only selected feature dimensions with a Gaussian distribution;
            :obj:`"onehot"`: mask all feature dimensions of the selected node with a one-hot vector.
            (default: :obj:`"whole"`)
        mask_ratio (float, optinal): The ratio of node attributes to be masked. (default: :obj:`0.1`)
        mask_mean (float, optional): Mean of the Gaussian distribution to generate masking values.
            (default: :obj:`0.5`)
        mask_std (float, optional): Standard deviation of the distribution to generate masking values. 
            Must be non-negative. (default: :obj:`0.5`)
    """

    def __init__(self, mode='whole', mask_ratio=0.1, mask_mean=0.5, mask_std=0.5, return_mask=False):
        self.mode = mode
        self.mask_ratio = mask_ratio
        self.mask_mean = mask_mean
        self.mask_std = mask_std
        self.return_mask = return_mask
    
    def do_trans(self, data):
        
        node_num, feat_dim = data.x.size()
        x = data.x.detach().clone()

        if self.mode == "whole":
            mask = torch.zeros(node_num)
            mask_num = int(node_num * self.mask_ratio)
            idx_mask = np.random.choice(node_num, mask_num, replace=False)
            x[idx_mask] = torch.tensor(np.random.normal(loc=self.mask_mean, scale=self.mask_std, 
                                                        size=(mask_num, feat_dim)), dtype=torch.float32)
            mask[idx_mask] = 1

        elif self.mode == "partial":
            mask = torch.zeros((node_num, feat_dim))
            for i in range(node_num):
                for j in range(feat_dim):
                    if random.random() < self.mask_ratio:
                        x[i][j] = torch.tensor(np.random.normal(loc=self.mask_mean, 
                                                                scale=self.mask_std), dtype=torch.float32)
                        mask[i][j] = 1

        elif self.mode == "onehot":
            mask = torch.zeros(node_num)
            mask_num = int(node_num * self.mask_ratio)
            idx_mask = np.random.choice(node_num, mask_num, replace=False)
            x[idx_mask] = torch.tensor(np.eye(feat_dim)[np.random.randint(0, feat_dim, size=(mask_num))], dtype=torch.float32)
            mask[idx_mask] = 1

        else:
            raise Exception("Masking mode option '{0:s}' is not available!".format(self.mode))

        if self.return_mask:
            return Data(x=x, edge_index=data.edge_index, mask=mask)
        else:
            return Data(x=x, edge_index=data.edge_index, name=data.name, edge_attr=data.edge_attr)
        
    def views_fn(self, data):
        """
        Method to be called when :class:`EdgePerturbation` object is called.
        
        Args:
            data (:class:`torch_geometric.data.Data`): The input graph or batched graphs.
            
        :rtype: :class:`torch_geometric.data.Data`.  
        """

        if isinstance(data, Batch):
            dlist = [self.do_trans(d) for d in data.to_data_list()]
            return Batch.from_data_list(dlist)
        elif isinstance(data, Data):
            return self.do_trans(data)

class GraphAugmenter:
    """
    Aggregater for graph augmentations.
    """
    def __init__(self, augment_list):
        self.augment_functions = []
        for augment in augment_list:
            if augment == "node_dropping":
                function = UniformSample()
            elif augment == "random_walk_subgraph":
                function = RWSample()
            elif augment == "node_attr_mask":
                function = NodeAttrMask()
            self.augment_functions.append(function)
            
    def get_positive_sample(self, current_graph):
        """
        Possible augmentations include the following:
            node_dropping()
            random_walk_subgraph()
            node_attr_mask()
        """

        graph_temp = current_graph
        for function in self.augment_functions:
            graph_temp = function.views_fn(graph_temp)
        return graph_temp
