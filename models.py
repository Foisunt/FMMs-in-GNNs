import torch
from torch import nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, GATConv
from torch.nn import Dropout, Linear, LayerNorm, BatchNorm1d, Identity
from torch_geometric.utils import dropout_adj

#https://discuss.pytorch.org/t/weight-decay-in-the-optimizers-is-a-bad-idea-especially-with-batchnorm/16994/2
# def add_weight_decay(model, settings):
#     decay = []
#     no_decay = []
#     for name, param in model.named_parameters():
#         if not param.requires_grad:
#             continue
#         if len(param.shape) == 1:
#             no_decay.append(param)
#         else:
#             decay.append(param)
#     return [
#         {'params': decay, 'weight_decay': settings["weight_decay"]},
#         {'params': no_decay, 'weight_decay': 0.}]

def add_weight_decayAll(model, settings):
    decay = []
    no_decay = []
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        if name[:3] == "bns":
            no_decay.append(param)
        else:
            decay.append(param)
    return [
        {'params': decay, 'weight_decay': settings["weight_decay"]},
        {'params': no_decay, 'weight_decay': 0.}]

# def create_wd_groups(self, settings):
#     decay = []
#     no_decay = []
#     for name, param in self.named_parameters():
#         if not param.requires_grad:
#             continue
#         if name == "convs.0.lin.weight":
#             decay.append(param)
#         else:
#             no_decay.append(param)
#     return [{'params': decay, 'weight_decay': settings["weight_decay"]},
#         {'params': no_decay, 'weight_decay': 0.}]
 

def get_feature_dis(x):
    """
    x :           batch_size x nhid
    x_dis(i,j):   item means the similarity between x(i) and x(j).
    """
    x_dis = x@x.T
    mask = torch.eye(x_dis.shape[0]).to(x.device)
    x_sum = torch.sum(x**2, 1).reshape(-1, 1)
    x_sum = torch.sqrt(x_sum).reshape(-1, 1)
    x_sum = x_sum @ x_sum.T
    x_dis = x_dis*(x_sum**(-1))
    x_dis = (1-mask) * x_dis
    return x_dis

    
class Model(torch.nn.Module):
    def __init__(self, in_channels, out_channels, hidden_channels, layer_kind, layer_num, layer_params, act_fn, settings):
        super().__init__()
        self.layers = torch.nn.ModuleList()
        self.norms = torch.nn.ModuleList()
        self.normF = {"id":lambda x: Identity(), "bn":lambda hd: BatchNorm1d(hd), "ln":lambda hd: LayerNorm(hd, eps=1e-6)}[settings["norm"]]
        self.act_fn = act_fn
        self.dropoutIN = Dropout(settings["drop_input"])
        self.dropoutM = Dropout(settings["drop_model"])
        self.dropAdj_p = settings["drop_adj"]
        self.emb_dims=[]
        #if settings["lbl_in"]:
        #    ch_in = in_channels+out_channels
        #else:
        ch_in = in_channels
        
        model_type = settings["model_type"]
        {"lin":self.lin_init, "res":self.res_init, "denseLin":self.denseLin_init}[model_type](ch_in, out_channels, hidden_channels, layer_kind, layer_num, layer_params, model_type) 
        self.fwd = {"lin":self.forward_lin, "res":self.forward_res, "denseLin":self.forward_denseLin}[model_type] 
        if settings["dataset"][:3]=="ppi":
            self.out_fn = lambda x : torch.sigmoid(x)
        else:
            self.out_fn = lambda x : F.log_softmax(x, dim=1)
        
        self.nc_l = int(settings["loss_NC"].split("@")[1])
        if self.nc_l <0:
            self.nc_l = layer_num + self.nc_l
        
        self.nc_w = float(settings["loss_NC"].split("@")[0])

        if self.nc_l==0:
            self.fwdL = self.fwd
        else:
            self.fwdL = {"lin":self.forward_linL, "res":self.forward_resL, "denseLin":self.forward_denseLinL}[model_type]
    
    @property
    def device(self):
        return next(self.parameters()).device

    # for normal init
    def lin_init(self, in_channels, out_channels, hidden_channels, layer_kind, layer_num, layer_params, model_type):
        fak = layer_params.get("heads", 1)
        self.norms.append(self.normF(in_channels))
        self.layers.append(layer_kind(in_channels, hidden_channels, **layer_params))
        for _ in range(layer_num - 2):
            self.norms.append(self.normF(hidden_channels*fak))
            self.emb_dims.append(hidden_channels*fak)
            self.layers.append(layer_kind(hidden_channels*fak, hidden_channels, **layer_params))
        if layer_kind == GATConv:
            layer_params["heads"] = 1
        self.norms.append(self.normF(hidden_channels*fak))
        self.emb_dims.append(hidden_channels*fak)
        self.layers.append(layer_kind(hidden_channels*fak, out_channels, **layer_params))
        self.emb_dims.append(out_channels)
        
    # for resnet init
    def res_init(self, in_channels, out_channels, hidden_channels, layer_kind, layer_num, layer_params, model_type):
        fak = layer_params.get("heads", 1)
        mtls = model_type.split("_")
        if mtls[0] == "res": hdim2 = hidden_channels
        #elif mtls[0] == "res2": hdim2 = int(mtls[1])
        
        self.norms.append(self.normF(in_channels))
        self.layers.append(layer_kind(in_channels, hidden_channels, **layer_params))
        for _ in range(int((layer_num - 2)/2)):#/2 is an artifact to support 2 blocks off axis, no longer in use
            self.norms.append(self.normF(hidden_channels*fak))
            self.emb_dims.append(hidden_channels*fak)
            self.layers.append(layer_kind(hidden_channels*fak, hdim2, **layer_params))
            self.norms.append(self.normF(hdim2*fak))
            self.emb_dims.append(hdim2*fak)
            self.layers.append(layer_kind(hdim2*fak, hidden_channels, **layer_params))
        self.norms.append(self.normF(hidden_channels*fak))
        self.emb_dims.append(hidden_channels*fak)
        self.layers.append(Linear(hidden_channels*fak, out_channels))
        self.emb_dims.append(out_channels)
    
    def denseLin_init(self, in_channels, out_channels, hidden_channels, layer_kind, layer_num, layer_params, model_type):
        fak = layer_params.get("heads", 1)
        growth = int(model_type.split("_")[1])
        self.norms.append(self.normF(in_channels))
        self.layers.append(Linear(in_channels, hidden_channels))
        for i in range(1, layer_num-1):
            self.norms.append(self.normF(hidden_channels + growth*(i-1)*fak))
            self.emb_dims.append(hidden_channels + growth*(i-1)*fak)
            self.layers.append(layer_kind(hidden_channels + growth*(i-1)*fak, growth, **layer_params))
        self.norms.append(self.normF(hidden_channels + growth*(layer_num-2)*fak))
        self.emb_dims.append(hidden_channels + growth*(layer_num-2)*fak)
        self.layers.append(Linear(hidden_channels + growth*(layer_num-2)*fak, out_channels))
        self.emb_dims.append(out_channels)
        
    
    def forward(self, data):
        if self.training:
            return self.fwdL(data)
        return self.fwd(data)
    
    def forward_lin(self, data):
        x, adj_t = data.feat, dropout_adj(data.edge, p = self.dropAdj_p)[0]
        x = self.layers[0](self.dropoutIN(self.norms[0](x)), adj_t)
        for i in range(1,len(self.layers)-1):
            x = self.layers[i](self.act_fn(self.dropoutM(self.norms[i](x))), adj_t)
        return self.out_fn(self.layers[-1](self.act_fn(self.dropoutM(self.norms[-1](x))), adj_t))
    
    def forward_res(self, data):
        x, adj_t = data.feat, dropout_adj(data.edge, p = self.dropAdj_p)[0]
        x = self.layers[0](self.dropoutIN(self.norms[0](x)), adj_t)
        for i in range(1, len(self.layers)-1):
            x = x+self.layers[i](self.act_fn(self.dropoutM(self.norms[i](x))), adj_t)
        return self.out_fn(self.layers[-1](self.act_fn(self.dropoutM(self.norms[-1](x)))))
    
    def forward_denseLin(self, data):
        x, adj_t = data.feat, dropout_adj(data.edge, p = self.dropAdj_p)[0]
        x = self.layers[0](self.dropoutIN(self.norms[0](x)))
        for i in range(1, len(self.layers)-1):
            x = torch.cat([x,self.layers[i](self.act_fn(self.dropoutM(self.norms[i](x))), adj_t)], -1)
        return self.out_fn(self.layers[-1](self.act_fn(self.dropoutM(self.norms[-1](x)))))
    
    def forward_linL(self, data):
        x, adj_t = data.feat, dropout_adj(data.edge, p = self.dropAdj_p)[0]
        x = self.layers[0](self.dropoutIN(self.norms[0](x)), adj_t)
        if self.nc_l == 0:
            out_nc = get_feature_dis(x)
        for i in range(1,len(self.layers)-1):
            x = self.layers[i](self.act_fn(self.dropoutM(self.norms[i](x))), adj_t)
            if self.nc_l == i:
                out_nc = get_feature_dis(x)
        return self.out_fn(self.layers[-1](self.act_fn(self.dropoutM(self.norms[-1](x))), adj_t)), out_nc
    
    def forward_resL(self, data):
        x, adj_t = data.feat, dropout_adj(data.edge, p = self.dropAdj_p)[0]
        x = self.layers[0](self.dropoutIN(self.norms[0](x)), adj_t)
        if self.nc_l == 0:
            out_nc = get_feature_dis(x)
        for i in range(1, len(self.layers)-1):
            x = x + self.layers[i](self.act_fn(self.dropoutM(self.norms[i](x))), adj_t)
            if self.nc_l == i:
                out_nc = get_feature_dis(x)
        return self.out_fn(self.layers[-1](self.act_fn(self.dropoutM(self.norms[-1](x))))), out_nc
    
    def forward_denseLinL(self, data):
        x, adj_t = data.feat, dropout_adj(data.edge, p = self.dropAdj_p)[0]
        x = self.layers[0](self.dropoutIN(self.norms[0](x)))
        if self.nc_l == 0:
            out_nc = get_feature_dis(x)     
        for i in range(1, len(self.layers)-1):
            x = torch.cat([x,self.layers[i](self.act_fn(self.dropoutM(self.norms[i](x))), adj_t)], -1)
            if self.nc_l == i:
                out_nc = get_feature_dis(x)
        return self.out_fn(self.layers[-1](self.act_fn(self.dropoutM(self.norms[-1](x))))), out_nc
    
    def create_wd_groups(self, settings):
        return add_weight_decayAll(self, settings)
    
class GCN(Model):
    def __init__(self, in_channels, out_channels, settings):
        #multi_lbl = settings["dataset"][:3]=="ppi"
        #print("gcnind", settings["inductive"])
        ind = ((settings["dataset"][:3]=="ppi") | settings.get("inductive", False))
        ls = [int(x) for x in settings["model"].split("_")[1:]]
        super().__init__(in_channels, out_channels, hidden_channels = ls[0], layer_kind = GCNConv, layer_num = ls[1],layer_params={"cached":not ind}, #only cache if the graph is fixed
                         act_fn = F.relu, settings = settings)
        
    # def create_wd_groups(self, settings):
    #     return add_weight_decayAll(self, settings)
    
    def create_wd_groups(self, settings):
        decay = []
        no_decay = []
        for name, param in self.named_parameters():
            if not param.requires_grad:
                continue
            if name == "layers.0.lin.weight":
                decay.append(param)
            else:
                no_decay.append(param)
        return [{'params': decay, 'weight_decay': settings["weight_decay"]},
            {'params': no_decay, 'weight_decay': 0.}]

        
class GAT(Model):
    def __init__(self, in_channels, out_channels, settings):
        #multi_lbl = settings["dataset"][:3]=="ppi"
        ls = [int(x) for x in settings["model"].split("_")[1:]]
        super().__init__(in_channels, out_channels, hidden_channels = ls[0], layer_kind = GATConv, layer_num = ls[1],layer_params={"heads":ls[2] , "dropout":settings["drop_attn"]}, 
                         act_fn = F.relu, settings = settings)

class GMLP(Model):
    def __init__(self, in_channels, out_channels, settings):
        #multi_lbl = settings["dataset"][:3]=="ppi"
        ls = [int(x) for x in settings["model"].split("_")[1:]]
        super().__init__(in_channels, out_channels, hidden_channels = ls[0], layer_kind = Linear, layer_num = ls[1],layer_params={}, 
                         act_fn = F.gelu, settings = settings)
        #self._init_weights()
        self.fwd = {"lin":self.forward_lin, "res":self.forward_res, "denseLin":self.forward_denseLin}[settings["model_type"]] 
        self.fwdL = {"lin":self.forward_linL, "res":self.forward_resL, "denseLin":self.forward_denseLinL}[settings["model_type"]]
        #print("gmlp", self.nc_l, len(self.layers))
        self.norms[-1] = LayerNorm(ls[0], eps=1e-6, elementwise_affine=False)
        
    def _init_weights(self):
        for l in self.layers:
            nn.init.xavier_uniform_(l.weight)
            nn.init.normal_(l.bias, std=1e-6)
    
    def forward(self, data):
        if self.training:
            return self.fwdL(data)
        return self.fwd(data.feat)
        
    def forward_lin(self, data):
        x = self.layers[0](self.dropoutIN(self.norms[0](data)))
        for i in range(1,len(self.layers)-1):
            x = self.layers[i](self.act_fn(self.dropoutM(self.norms[i](x))))
        return self.out_fn(self.layers[-1](self.act_fn(self.dropoutM(self.norms[-1](x)))))
    
    def forward_res(self, data):
        x = self.layers[0](self.dropoutIN(self.norms[0](data)))
        for i in range(1, len(self.layers)-1):
            x = x + self.layers[i](self.act_fn(self.dropoutM(self.norms[i](x))))
        return self.out_fn(self.layers[-1](self.act_fn(self.dropoutM(self.norms[-1](x)))))
        return self.out_fn(self.layers[-1](self.act_fn(self.dropoutM(x))))

    def forward_denseLin(self, data):
        x = self.layers[0](self.dropoutIN(self.norms[0](data)))
        for i in range(1, len(self.layers)-1):
            x = torch.cat([x,self.layers[i](self.act_fn(self.dropoutM(self.norms[i](x))))], -1)
        return self.out_fn(self.layers[-1](self.act_fn(self.dropoutM(self.norms[-1](x)))))
        
    def forward_linL(self, data):
        x = self.layers[0](self.dropoutIN(self.norms[0](data)))
        if self.nc_l == 0:
            out_nc = get_feature_dis(x)
        for i in range(1,len(self.layers)-1):
            x = self.layers[i](self.act_fn(self.dropoutM(self.norms[i](x))))
            if self.nc_l == i:
                out_nc = get_feature_dis(x)
        return self.out_fn(self.layers[-1](self.act_fn(self.dropoutM(self.norms[-1](x))))), out_nc
    
    def forward_resL(self, data):
        x = self.layers[0](self.dropoutIN(self.norms[0](data)))
        if self.nc_l == 0:
            out_nc = get_feature_dis(x)
        for i in range(1, len(self.layers)-1):
            x = x + self.layers[i](self.act_fn(self.dropoutM(self.norms[i](x))))
            if self.nc_l == i:
                out_nc = get_feature_dis(x)
        #return self.out_fn(self.layers[-1](self.act_fn(self.dropoutM(self.norms[-1](x))))), out_nc
        return self.out_fn(self.layers[-1](self.act_fn(self.dropoutM(x)))), out_nc
        
    def forward_denseLinL(self, data):
        x = self.layers[0](self.dropoutIN(self.norms[0](data)))
        if self.nc_l == 0:
            out_nc = get_feature_dis(x)     
        for i in range(1, len(self.layers)-1):
            x = torch.cat([x,self.layers[i](self.act_fn(self.dropoutM(self.norms[i](x))))], -1)
            if self.nc_l == i:
                out_nc = get_feature_dis(x)
        return self.out_fn(self.layers[-1](self.act_fn(self.dropoutM(self.norms[-1](x))))), out_nc

       
#returns an initialized model
def get_model(settings, num_features, num_classes):
    name_d = {
        "GCN_":GCN,
        "G1T_":GAT,
        "GMLP":GMLP,
    }
    m = name_d[settings["model"][:4]]
   
    
    #if settings["model"][:3] == "GC2":
    #    return m(num_features, num_classes, *[int(x) for x in settings["model"].split("_")[1:]], dropout = settings["drop_rate"], alpha = settings["alpha"], theta = settings["theta"], multi_lbl=multi_lbl)
    
    return m(num_features, num_classes, settings)
    