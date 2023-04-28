import torch
import torch_sparse
import torch.nn.functional as F
from torch import nn
import numpy as np
from sam import SAM
from pathlib import Path
import pandas as pd
from functools import partial
from torch.nn.utils import parameters_to_vector as p2v
from torch.nn.utils import vector_to_parameters as v2p
from torchmetrics.functional.classification import multilabel_f1_score
from time import perf_counter

from ES import EarlyStoppingA, EarlyStoppingL
from utils import *

import copy
import time
from datetime import datetime


def Ncontrast(x_dis, adj_label, tau):
    """
    compute the Ncontrast loss
    """
    x_dis = torch.exp( tau * x_dis)# div by tau in paper ?
    x_dis_sum = torch.sum(x_dis, 1)
    x_dis_sum_pos = torch.sum(x_dis*adj_label, 1)
    loss = -torch.log(x_dis_sum_pos * (x_dis_sum**(-1))+1e-8).mean()
    return loss

#squeeze 1st dim but keep 0th dim (batch size = dim 0 might be 1)
def squeeze1if2(t):
    if len(t.shape)==1:
        return t 
    else:
        return t.squeeze(1)

# def lossfin(outNC, outCE, lbls ,yij, tau, alpa, cl_loss_fn):
#     loss_train_class = cl_loss_fn(outCE, squeeze1if2(lbls))
#     loss_Ncontrast = Ncontrast(outNC, yij, tau)
#     return loss_train_class + loss_Ncontrast * alpa
        
def loss1(f, o, lbls, mask, **unused):
    #if torch.isnan(o).sum()>0:
    #    raise Exception("nan will kill cuda")
    return f(o[mask], squeeze1if2(lbls[mask])) #this line sometimes tiggerst cuda assert errors on ppi+gat, specifically the croseentropy call in that case
    
def loss3(f, o, lbls, mask, w_nc, adj_label = None, tau=None):
    l_label = f(o[0][mask], squeeze1if2(lbls[mask]))
    l_nc = 0
    if w_nc>0:
        l_nc = w_nc*Ncontrast(o[1], adj_label, tau)
    return l_label+l_nc


def accuracy(output, labels):
    preds = output.argmax(dim = 1).type_as(labels)
    correct = preds.eq(labels).double()
    correct = correct.sum()
    return correct / len(labels)

def val_(data, model, loss_fn, _):
    model.eval()
    with torch.no_grad():
        lbls = torch.cat([x[0].lbl[x[1]] for x in data],axis=0)
        output = torch.cat([model(x[0])[x[1]] for x in data],axis=0)
        
        loss = loss_fn(output, squeeze1if2(lbls))
        acc = accuracy(output, squeeze1if2(lbls))
        #out = model(data)
        #loss = loss_fn(out[mask], squeeze1if2(data.lbl)[mask])
        #acc = accuracy(out[mask], squeeze1if2(data.lbl)[mask])
    return (float(loss), float(acc))

#todo gmlp ar, xnorm

def val_F1(data, model, loss_fc, threshold, ncl = 121):
    loss_val = 0
    f1_micro, f1_macro, f1_weighted, f1_sample = 0, 0, 0, 0
    model.eval()
    with torch.no_grad():
        
        lbls = torch.cat([x[0].lbl[x[1]] for x in data],axis=0)
        output = torch.cat([model(x[0])[x[1]] for x in data],axis=0)
        
        loss_val = loss_fc(output, lbls)
        
        lbls = lbls.int()
        
        f1_micro = multilabel_f1_score(output, lbls, num_labels = ncl, threshold=threshold, average='micro')
        f1_macro = multilabel_f1_score(output, lbls, num_labels = ncl, threshold=threshold, average='macro')
        f1_weighted = multilabel_f1_score(output, lbls, num_labels = ncl, threshold=threshold, average='weighted')
        #f1_sample = multilabel_f1_score(output, lbls, num_classes = ncl, threshold=threshold, average='samples') #no longer supported by torchmetrics
        
        
    return loss_val.item(), {"f1_micro":f1_micro.item(), "f1_macro":f1_macro.item(), "f1_weighted":f1_weighted.item(), "f1_sample": 0} #f1_sample.item()}


class TrainHandler():
    def __init__(self, settings):
        self.loss_fn = torch.nn.NLLLoss()
        self.valf = val_
        self.threshold = 0.5
        self.ppi = False
        if settings["dataset"].split("_")[0]=="ppi":
            self.ppi = True
            self.loss_fn = torch.nn.BCELoss()
            self.valf = val_F1
            self.threshold = settings["threshold"]
        self.ani = True if settings["ani"] != False else False
        self.ani_sig = settings["ani"] if self.ani else 0
        self.oldE = 0
        self.lossSum = loss1
        
        self.saf_outs = []
        self.saf_safe = False
        self.saf_do = False
        if settings["saf_eps"] != False:
            self.saf_lt =  np.log(settings["saf_tau"])
            self.saf_lambda =  settings["saf_lambda"]
        
    def train_epoch(self, data, model, optimizer):
        model.train()
        if self.saf_do:
            old_preds = self.saf_outs.pop(0)
        new_preds = []
        
        lo = 0
        for graph, mask in data.getTrain():
            optimizer.zero_grad()
            out = model(graph)
            loss = self.lossSum(self.loss_fn, out, graph.lbl, mask) #this line sometimes tiggerst cuda assert errors on ppi+gat
            if self.saf_safe:
                if self.ppi: #ppi has sigmoid output, others log softmax
                    out = out.log()
                out = out-self.saf_lt
                new_preds.append(out.data)
                if self.saf_do:
                    loss = loss + self.saf_lambda*F.kl_div(out, old_preds.pop(0), log_target=True, reduction='batchmean')#torchs kl loss swapps the input compared to kl definition to be consistent with other torch losses
            lo += float(loss)
            loss.backward()
            #torch.nn.utils.clip_grad_norm_(model.parameters(), 1)#tested for gat+ppi stability issues          
            optimizer.step()
            loss = 0

        if self.saf_safe:
            self.saf_outs.append(new_preds)
        
        if self.ani == True:
            self.do_ani(model)
        return float(lo)

    def do_ani(self, model):
        vec = p2v(model.parameters())
        tmp = torch.randn_like(vec)*self.ani_sig
        vec = vec + tmp - self.oldE
        self.oldE = tmp
        v2p(vec, model.parameters())
    def set_ani(self,v):
        self.ani = v
    def set_saf_safe(self,v):
        self.saf_safe = v
    def set_saf_do(self,v):
        self.saf_do = v
    def get_ES(self):
        return EarlyStoppingL
    def val(self, data, model):
        return self.valf(data, model, self.loss_fn, self.threshold)
    
    
class TrainHandlerGMLP(TrainHandler):
    def __init__(self, settings):
        super().__init__(settings)
        self.B = settings["b"]
        self.tau = settings["tau"]
        self.nc = float(settings["loss_NC"].split("@")[0])
        self.lossSum = loss3
        
    def train_epoch(self, data, model, optimizer):            
        model.train()
        if self.saf_do:
            old_preds = self.saf_outs.pop(0)
        new_preds = []
        
        lo = 0
        for graph, mask in data.getTrain():
            optimizer.zero_grad()
            if self.B != 1:
                batch_size = self.B if self.B > 1 else int(self.B*len(graph.lbl))
                #if batch_size > 30000: #rough vram limit of 40 GB for Ar + model
                #    print("batch size of", str(batch_size), "with B =", str(self.b), "and len(data.y) =", str(len(data.y)))
                
                idB = torch.randperm(graph.feat.size(0))[:batch_size]
                msk = mask[idB]
                out = model(graph.feat[idB])
                A_smpl = graph.edge[idB,:][:, idB]#>99% epoch time; 126s vs sum=1s for arxiv, b=20k, sparse
                if type(A_smpl) == torch_sparse.tensor.SparseTensor:
                    A_smpl = A_smpl.to_dense()
                A_smpl = A_smpl.to(graph.feat.device)
                loss = self.lossSum(self.loss_fn, out, graph.lbl[idB], msk, w_nc = self.nc, adj_label = A_smpl, tau=self.tau)
            else:
                out = model(graph.feat)
                loss = self.lossSum(self.loss_fn, out, graph.lbl, mask, w_nc = self.nc, adj_label = graph.edge, tau=self.tau)
            
            if self.saf_safe:
                out = out[0] #out[1] is the nc sim matrix
                if self.ppi: #ppi has sigmoid output, others log softmax
                    out = out.log()
                out = out-self.saf_lt
                new_preds.append(out.data)
                if self.saf_do:
                    loss = loss + self.saf_lambda*F.kl_div(out, old_preds.pop(0), log_target=True, reduction='batchmean')#torchs kl loss swapps the input compared to kl definition to be consistent with other torch losses
            lo += float(loss)
            loss.backward()
            optimizer.step()
            
        if self.saf_safe:
            self.saf_outs.append(new_preds)
        if self.ani == True:
            self.do_ani(model)
            
        return float(lo)
    def get_ES(self):
        return EarlyStoppingA
#    def val(self, data, model):
#        return self.valf(data, model, self.loss_fn, self.threshold)
    

class TrainHandlerSAM(TrainHandler):
    def __init__(self, settings):
        super().__init__(settings)
        self.sam_mod = "nomod"
        self.a = float("nan")
        if settings.get("sam_mod", "nomod") != "nomod":
            tmp = settings["sam_mod"].split("_")
            self.sam_mod = tmp[0]
            self.a = float(tmp[1])
        if settings["saf_eps"] != False:
            raise NotImplementedError("sam+saf is not implemented")

    def train_epoch(self, data, model, optimizer):
        model.train()
        
        lo = 0
        lA = 0
        
        for graph, mask in data.getTrain():
            enable_running_stats(model)
            optimizer.zero_grad()
            out = model(graph)
            loss = self.lossSum(self.loss_fn, out, graph.lbl, mask)
            loss.backward()
            optimizer.first_step()#zero_grad=True)
            
            if self.sam_mod == "pgn":
                for p in model.parameters(): #penalizing grad norm, space optimized to run in place
                    p.grad *= (1-self.a)/self.a
            elif self.sam_mod == "gsam":
                df = grads_to_vec(model.parameters()) #normal gradient
                    
            
            disable_running_stats(model)
            lossAdv = self.lossSum(self.loss_fn, model(graph), graph.lbl, mask)
            lossAdv.backward()
            
            if self.sam_mod == "pgn":
                for p in model.parameters():
                    p.grad *= self.a
            elif self.sam_mod == "gsam":
                dfp = grads_to_vec(model.parameters()) #perturbed gradient
                df = df-projAonB(df, dfp) #h; vertical component of df 
                vec_to_grads(dfp-self.a*df, model.parameters())
                
            optimizer.second_step(zero_grad=True)
            
            lo += loss.item()
            lA += lossAdv.item()
            
        if self.ani == True:      
            self.do_ani(model)
            
        return float((1-self.a)*lo + self.a*lA)
    
    

class TrainHandlerGMLP_SAM(TrainHandlerGMLP, TrainHandlerSAM):
    def __init__(self, settings):
        super().__init__(settings) #calls both gmlps and sams inits
        if settings["saf_eps"] != False:
            raise NotImplementedError("gmlp: sam+saf is not implemented")

    def train_epoch(self, data, model, optimizer):            
        model.train()
        lo = 0
        lA = 0
        batch_size, idB, msk, A_smlp = None, None, None, None
        
        for graph, mask in data.getTrain():
            enable_running_stats(model)
            optimizer.zero_grad()
            
            if self.B != 1:
                batch_size = self.B if self.B > 1 else int(self.B*len(graph.lbl))
                
                idB = torch.randperm(graph.feat.size(0))[:batch_size]
                msk = mask[idB]
                out = model(graph.feat[idB])
                A_smpl = graph.edge[idB,:][:, idB]
                if type(A_smpl) == torch_sparse.tensor.SparseTensor:
                    A_smpl = A_smpl.to_dense()
                A_smpl = A_smpl.to(graph.feat.device)
                loss = self.lossSum(self.loss_fn, out, graph.lbl[idB], msk, w_nc = self.nc, adj_label = A_smpl, tau=self.tau)
            else:
                out = model(graph.feat)
                loss = self.lossSum(self.loss_fn, out, graph.lbl, mask, w_nc = self.nc, adj_label = graph.edge, tau=self.tau)

            loss.backward()
            optimizer.first_step()
            if self.sam_mod == "pgn":
                for p in model.parameters(): #penalizing grad norm, space optimized to run in place
                    p.grad *= (1-self.a)/self.a
            elif self.sam_mod == "gsam":
                df = grads_to_vec(model.parameters()) #normal gradient
                #df = grads_to_vec(model) #normal gradient
            
            disable_running_stats(model)
            if self.B != 1:
                out = model(graph.feat[idB])
                lossAdv = self.lossSum(self.loss_fn, out, graph.lbl[idB], msk, w_nc = self.nc, adj_label = A_smpl, tau=self.tau)
            else:
                out = model(graph.feat)
                lossAdv = self.lossSum(self.loss_fn, out, graph.lbl, mask, w_nc = self.nc, adj_label = graph.edge, tau=self.tau)
                
            lossAdv.backward()            
            if self.sam_mod == "pgn":
                for p in model.parameters():
                    p.grad *= self.a
            elif self.sam_mod == "gsam":
                dfp = grads_to_vec(model.parameters()) #perturbed gradient
                df = df-projAonB(df, dfp) #h; vertical component of df 
                vec_to_grads(dfp-self.a*df, model.parameters())
            optimizer.second_step(zero_grad=True)

            lo += loss.item()
            lA += lossAdv.item()
            
        if self.ani == True:      
            self.do_ani(model) 
            
        return float((1-self.a)*lo + self.a*lA)
            
            
def eval_m(d, m, tH):
    m.eval()
    trl, tra = tH.val(d.getTrain(), m)
    vl, va = tH.val(d.getVal(), m)
    tel, tea = tH.val(d.getTest(), m)
    return [trl, tra, vl, va, tel, tea]
    
def train(data, model, optimizer, settings, save_path, trainHandler):
    tH = trainHandler(settings)
    if settings["saf_eps"] != False:
        Estart = int(settings["saf_eps"].split("_")[0])
        Ediff = int(settings["saf_eps"].split("_")[1])
    wa = False
    if settings["early_stopping"] != False:
        ES = tH.get_ES()(settings["early_stopping"], True, save_path)
        wa = settings.get("wa", False)
        
        if wa != False:
            #if settings["norm"]=="bn":
            #    raise NotImplementedError("wa+bn")
            ES.init_wa(wa, model)
    end = -1
    t0 = perf_counter()
    for epoch in range(settings["epochs"]):
        if settings["ani"] != False:
            if epoch == settings["aniend"]:
                tH.set_ani(False)
        
        if settings["saf_eps"] != False:
            if epoch == (Estart-Ediff):
                tH.set_saf_safe(True)
            if epoch == Estart:
                tH.set_saf_do(True)
        #try:
        l = tH.train_epoch(data, model, optimizer)
        #except Exception as e:
        #    print("Training died in epoch", epoch, "eval and next seed")
        #    break
        # if epoch%250 ==0:
        #     now = datetime.now()
        #     current_time = now.strftime("%H:%M:%S")
        #     print("epoch ", epoch, "Current Time =", current_time)
        #    torch.cuda.empty_cache()
        if (settings["early_stopping"] != False):
            #with torch.no_grad():
            val_loss, val_acc = tH.val(data.getVal(), model)
            stop = ES(val_loss, val_acc, model)
            if wa!=False:
                ES.update_wa(model)#copy.deepcopy(model))
            if stop:
                #print("normal stop")
                end = epoch - settings["early_stopping"]
                break
    t_train = [perf_counter() -t0]
    if end == -1: 
        end = settings["epochs"]
    #load best model if ES is used
    if settings["early_stopping"] != False:
        model.load_state_dict(torch.load(str(save_path)))
    #torch.cuda.empty_cache()
    train_loss, train_acc, val_loss, val_acc, test_loss, test_acc = [], [], [], [], [], []
    t1 = perf_counter()
    eval_ls = eval_m(data, model, tH)  
    t_inf = [perf_counter() -t1]
    end = [end]
    wa_log = ["nowa"]
    wa_res = []
    if wa!= False:
        #print("val wa")
        sets = settings["wa"].split(" ")
        for s in sets:
            #torch.cuda.empty_cache()
            model = ES.get_wa(model, s)
            if settings["norm"]=="bn":
                model.train()
                with torch.no_grad():
                    for graph, mask in data.getTrain():
                        out = model(graph)
            t2 = perf_counter()    
            wa_res.append(eval_m(data, model, tH))
            t_inf.append(perf_counter() -t2)
            end.append(end[0]+int(s.split("_")[3]))
            t_train.append(t_train[0])
            wa_log.append(s)
    if wa!=False:
        eval_ls = list(zip(eval_ls, *wa_res))
    else:
        eval_ls = list(zip(eval_ls))
    df = pd.DataFrame(data={"wa":wa_log, "trained_epochs":end,"train_time":t_train, "inf_time":t_inf, "train_loss":eval_ls[0], "train_acc":eval_ls[1], "val_loss":eval_ls[2], "val_acc":eval_ls[3], "test_loss":eval_ls[4], "test_acc":eval_ls[5]})
    
    if (settings["early_stopping"] != False) and (settings["save_models"] == False):
        save_path.unlink()
        if wa:
            ES.cleanup_wa()
    #torch.cuda.empty_cache()
    return df
        

def get_train(settings):
    if settings["sam"]=="nosam":
        if settings["model"][:3]=="GML":
            tH = TrainHandlerGMLP
        else:
            tH = TrainHandler
    elif settings["sam"][:4]=="sam_" or settings["sam"][:4]=="asam":
        if settings["model"][:3]=="GML":
            tH = TrainHandlerGMLP_SAM
        else:
            tH = TrainHandlerSAM
    return partial(train, trainHandler=tH)        
