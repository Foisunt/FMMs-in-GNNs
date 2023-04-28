from torch_geometric.datasets import Planetoid, PPI, Amazon
from ogb.nodeproppred import PygNodePropPredDataset
import torch_geometric.transforms as T

import torch
import torch_sparse
import numpy as np
import scipy.sparse as sp

from torch_geometric.nn.conv.gcn_conv import gcn_norm
from torch_geometric.utils import to_dense_adj
from torch_sparse import SparseTensor
from torch_geometric.utils import subgraph

class Dataset:
    def __init__(self, settings, dev="cpu"):
        self.sett = settings
        ds = settings["dataset"]
        self.name = name = ds.split("_")[0] #name of the dataset
        self.split = ds.split("_")[1] #name of the train val test split
        self.dev = dev
        self.graphs = None #list (mostly with 1 elem) of "Data" containing the graphs
        self.train_graphs = None #list with (int) train graph idx
        self.val_graphs = None # '' val
        self.test_graphs = None # '' test
        self.train_nodes = None # list of (bool) training idx tensors for each graph
        self.val_nodes = None # '' val
        self.test_nodes = None # '' test
        self.f = lambda x : None # get different split, only plantoid datasets with rand or 622 splits
        self.num_node_features = None
        self.num_classes = None
        self.force_inductive = settings.get("inductive", False)
        transform = T.Compose([T.AddSelfLoops(), T.ToUndirected()]) if settings["transform_ds"] else T.Compose([])
        
        if self.name in ["Citeseer", "Cora", "PubMed", "Computers", "Photo"]:
            if self.name in ["Computers", "Photo"]:
                tmp = Amazon(root='dataset/', name=name)
            else:
                tmp = Planetoid(root='dataset/', name=name)
            
            self.num_node_features = tmp.num_node_features
            self.num_classes = tmp.num_classes
            self.graphs = [Data(transform(tmp[0]), self.dev, settings)]
            self.train_graphs, self.val_graphs, self.test_graphs = [0],[0],[0]
            if self.split == "planetoid":        
                self.train_nodes, self.val_nodes, self.test_nodes = [tmp[0].train_mask],[tmp[0].val_mask],[tmp[0].test_mask]
            elif self.split == "rand-pl":
                d = torch.load("dataset/"+name+"/own_pl_splits.pt")
                self.f = lambda x: d[x]
            elif self.split == "rand-622":
                d = torch.load("dataset/"+name+"/own_622_splits.pt")
                self.f = lambda x: d[x]
            else:
                raise NotImplementedError("Split "+str(split)+ " not implemented")
        
        elif self.name == "ogbn-arxiv":
            tmp = PygNodePropPredDataset(root = 'dataset/', name="ogbn-arxiv")
            self.num_node_features = tmp.num_node_features
            self.num_classes = tmp.num_classes
            self.graphs = [Data(transform(tmp[0]), self.dev, settings)]
            self.train_graphs, self.val_graphs, self.test_graphs = [0],[0],[0]
            
            self.train_nodes = [torch.zeros(tmp[0].x.shape[0], dtype = bool)]
            self.val_nodes = [torch.zeros(tmp[0].x.shape[0], dtype = bool)]
            self.test_nodes = [torch.zeros(tmp[0].x.shape[0], dtype = bool)]
            self.train_nodes[0][tmp.get_idx_split()["train"]] = True
            self.val_nodes[0][tmp.get_idx_split()["valid"]] = True
            self.test_nodes[0][tmp.get_idx_split()["test"]] = True
            
        elif self.name == "ppi":
            train = PPI(root='dataset/', split = "train")
            val = PPI(root='dataset/', split = "val")
            test = PPI(root='dataset/', split = "test")
            self.num_node_features = train.num_node_features
            self.num_classes = train.num_classes
            self.graphs, self.train_nodes, self.val_nodes, self.test_nodes = [], [], [None]*len(train), [None]*(len(train)+len(val))
            self.train_graphs = [x for x in range(len(train))]
            for g in train:
                self.graphs.append(Data(transform(g), self.dev, settings))
                self.train_nodes.append(torch.ones(g.x.shape[0], dtype = bool))
            self.val_graphs = [x for x in range(len(train), len(train)+len(val))]
            for g in val:
                self.graphs.append(Data(transform(g), self.dev, settings))
                self.val_nodes.append(torch.ones(g.x.shape[0], dtype = bool))
            self.test_graphs = [x for x in range(len(train)+len(val), len(train)+len(val)+len(test))]
            for g in test:
                self.graphs.append(Data(transform(g), self.dev, settings))
                self.test_nodes.append(torch.ones(g.x.shape[0], dtype = bool))
        else:
            raise NotImplementedError("Unknown Dataset: "+str(self.name))
                    
    def __repr__(self):
        return "Dataset("+self.name+") Instance with " +str(len(self.graphs))+" graph(s)" 
    
    # updates split in accordance with statrep
    def setRep(self, statrep):
        if self.name in ["Citeseer", "Cora", "PubMed", "Computers", "Photo"]:
            if self.split != "planetoid":
                self.train_nodes = [self.f(statrep)["train_mask"]]
                self.val_nodes = [self.f(statrep)["val_mask"]]
                self.test_nodes = [self.f(statrep)["test_mask"]]
        
        if self.force_inductive == True:
            self.train_ls = []
            for i in self.train_graphs:
                A_dev = self.graphs[i].edge.device
                train_ix = torch.logical_not(self.val_nodes[i]+self.test_nodes[i])#unused nodes in planetoid
                d = Data(self.graphs[i].feat[train_ix], self.graphs[i].lbl[train_ix], (subgraph(subset = train_ix.to("cpu"), edge_index = self.graphs[i].edge.to("cpu"), relabel_nodes=True)[0]).to(A_dev), self.dev, self.sett)
                self.train_ls.append((d, self.train_nodes[i][train_ix]))
            self.val_ls = []
            for i in self.val_graphs:
                A_dev = self.graphs[i].edge.device
                val_ix = torch.logical_not(self.test_nodes[i])#unused nodes in planetoid
                self.val_ls.append((Data(self.graphs[i].feat[val_ix], self.graphs[i].lbl[val_ix], (subgraph(subset = val_ix.to("cpu"), edge_index = self.graphs[i].edge.to("cpu"), relabel_nodes=True)[0]).to(A_dev), self.dev, self.sett), 
                                    self.val_nodes[i][val_ix]))
            
            self.test_ls = [(self.graphs[i], self.test_nodes[i]) for i in self.test_graphs]
        
        else:
            self.train_ls = [(self.graphs[i], self.train_nodes[i]) for i in self.train_graphs]
            self.val_ls = [(self.graphs[i], self.val_nodes[i]) for i in self.val_graphs]
            self.test_ls = [(self.graphs[i], self.test_nodes[i]) for i in self.test_graphs]
    
    

    #returns a list of graphs and a list of node idx    
    def getTrain(self):
        return self.train_ls
    def getVal(self):
        return self.val_ls
    def getTest(self):
        return self.test_ls

def row_normalize(mx):
    """Row-normalize sparse matrix"""
    rowsum = np.array(mx.sum(1))
    r_inv = np.power(rowsum, -1).flatten()
    r_inv[np.isinf(r_inv)] = 0.
    r_mat_inv = sp.diags(r_inv)
    mx = r_mat_inv.dot(mx)
    return mx

def getAr(data, r,ds, trans, device, b):
    tmp = None
    if ds == "ogbn-arxiv":
        if r==2 or r==3:
            if r==3 and trans==True:
                tmp = torch.load("dataset/ogbn_arxiv/A3_undir_dense.pt")
            else:
                t = {True:"undir", False:"dir"}[trans]
                pth = "dataset/ogbn_arxiv/A"+str(r)+"_"+t+"_sparse.pt"
                tmp = torch.load(pth)
    if tmp == None:
        nnodes = data.y.shape[0]
        edge_index, edge_weight = gcn_norm(data.edge_index, edge_weight=None, num_nodes=nnodes)
        A = SparseTensor(row=edge_index[0], col=edge_index[1], value=edge_weight, sparse_sizes=(nnodes, nnodes)).to(device)
        
        tmp = A
        torch.cuda.empty_cache()
        for i in range(r-1):
            tmp = tmp.matmul(A)
        del edge_index
        del edge_weight
        del A
    if type(tmp) == torch_sparse.tensor.SparseTensor:
        if tmp.density() > 0.01 or b==1:
            tmp = tmp.to_dense()
    
    torch.cuda.empty_cache()
    return tmp

def getAr2(y, e, r,ds, trans, device, b):
    tmp = None
    if ds == "ogbn-arxiv":
        raise NotImplementedError
    if tmp == None:
        nnodes = y.shape[0]
        edge_index, edge_weight = gcn_norm(e, edge_weight=None, num_nodes=nnodes)
        A = SparseTensor(row=edge_index[0], col=edge_index[1], value=edge_weight, sparse_sizes=(nnodes, nnodes)).to(device)
        tmp = A
        torch.cuda.empty_cache()
        for i in range(r-1):
            tmp = tmp.matmul(A)
        del edge_index
        del edge_weight
        del A
    if type(tmp) == torch_sparse.tensor.SparseTensor:
        if tmp.density() > 0.01 or b==1:
            tmp = tmp.to_dense()
    torch.cuda.empty_cache()
    return tmp
    
class Data:
    def __init__(self, a, b, c, dev = None, se=None):
        if dev == None:
            self.init1(a, b, c)
        else:
            self.init2(a,b,c,dev, se)
    
    def init1(self, gr, dev, settings):
        self.lbl = gr.y.to(dev)
        if len(self.lbl.shape)==1:
            self.lbl = self.lbl.unsqueeze(1)
        if settings["model"][:4] != "GMLP":
            self.feat = gr.x.to(dev)
            self.edge = gr.edge_index.to(dev)
        else:
            self.feat = torch.from_numpy(row_normalize(gr.x)).to(dev)
            if settings.get("inductive"): #ar depends on train split (if r>2)
                self.edge = gr.edge_index.to(dev)
            else:    
                self.edge = getAr(gr, settings["r"], settings["dataset"].split("_")[0], settings["transform_ds"], dev, settings["b"])
        self.feat_size= self.feat.shape[1]
        
    def init2(self, x, y, e, dev, settings):
        self.lbl = y.to(dev)
        if len(self.lbl.shape)==1:
            self.lbl = self.lbl.unsqueeze(1)
        if settings["model"][:4] != "GMLP":
            self.feat = x.to(dev)
            self.edge = e.to(dev)
        else:
            self.feat = x.to(dev)#rownorm already done, 
            self.edge = getAr2(y,e, settings["r"], settings["dataset"].split("_")[0], settings["transform_ds"], dev, settings["b"])
        self.feat_size= self.feat.shape[1]
    
        
    def __repr__(self):
        try:
            return "Data(feats: [{},{}], edges:[{},{}], lbls: [{},{}])".format(self.feat.shape[0], self.feat.shape[1], self.edge.shape[0], self.edge.shape[1], self.lbl.shape[0], self.lbl.shape[1])
        except:
            return "Data(feats: [{},{}], edges:[?,?], lbls: [{},{}])".format(self.feat.shape[0], self.feat.shape[1], self.lbl.shape[0], self.lbl.shape[1])
    def add_labels(self):
        raise NotImplementedError("add labels not implemented")
    def gmlp_xnorm(self):
        pass
    def gmlp_ar(self,r):
        pass
    
def get_dataset(settings, dev):
    return Dataset(settings, dev)

