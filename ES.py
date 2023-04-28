import torch
from torch.nn.utils import parameters_to_vector as p2v
from torch.nn.utils import vector_to_parameters as v2p
from torch.optim.swa_utils import AveragedModel
from pathlib import Path
import numpy as np
from collections import OrderedDict

from functools import partial
from time import perf_counter


def swa_fn(averaged_model_parameter, model_parameter, num_averaged):
    return averaged_model_parameter + (model_parameter - averaged_model_parameter) / (num_averaged + 1)
def ewa_fn(a, averaged_model_parameter, model_parameter, num_averaged):
    return a * averaged_model_parameter + (1-a) * model_parameter


class ES:
    def __init__(self, patience, save, path):
        self.patience = patience
        self.path = path
        self.save = save
        self.counter = 0
        
        self.swa_starts = []
        self.swa_ends = []
        self.swa_models = []
        self.swa_settings = []
        
        self.timeavg = 0
        
    def __call__():
        pass
    def init_wa(self, swa, model):
        new_s = swa.split(" ")
        self.swa_settings.extend(new_s)
        for i in range(len(new_s)):
            setting = new_s[i]
            params = setting.split("_")
            self.swa_starts.append(int(params[1]))
            self.swa_ends.append(int(params[3]))
            if params[0] == "s":
                self.swa_models.append(AveragedModel(model, avg_fn = swa_fn))#qf
#                self.swa_models.append(AveragedModel(model, device="cpu", avg_fn = swa_fn))#qf
            elif params[0] == "e":
                a = float(params[2])
                assert 0<a<1, "bad ewa alpha"
                self.swa_models.append(AveragedModel(model, avg_fn = partial(ewa_fn, a )))#qf
#                self.swa_models.append(AveragedModel(model, device="cpu", avg_fn = partial(ewa_fn, a )))#qf

            else:
                raise NotImplementedError("typ not in s, e")
    
    def update_wa(self, model):
 #       d = model.device
        #t0 = perf_counter()
#        model.to("cpu") #qf
        for i in range(len(self.swa_settings)):
            if self.swa_starts[i]<=0:
#                self.swa_models[i].to(model.device)
                self.swa_models[i].update_parameters(model)
#                self.swa_models[i].to("cpu")
                if self.counter == self.swa_ends[i]:
                    self.save_wa(self.swa_models[i], self.swa_settings[i])
            else:
                self.swa_starts[i]-=1
#        model.to(d)
        
        #td = perf_counter()-t0
        #self.timeavg = 0.9*self.timeavg+0.1*td
        #print("update took", td, "s, avg is", self.timeavg)
        
    def save_wa(self, model, wa_str):
        p = str(self.path)[:-3]+"_"+wa_str.replace(".", "+")+".pt"
        torch.save(model.state_dict(), p)

    def get_wa(self, model, wa_str):
        p = str(self.path)[:-3]+"_"+wa_str.replace(".", "+")+".pt"
                
        state_dict = torch.load(p)
        # create new OrderedDict that does not contain `module.`
        new_state_dict = OrderedDict()
        for k, v in state_dict.items():
            if k != "n_averaged":
                name = k[7:] # remove `module.`
                new_state_dict[name] = v
        model.load_state_dict(new_state_dict)
        return model
    
    def cleanup_wa(self):
        for s in self.swa_settings:
            Path(str(self.path)[:-3]+"_"+s.replace(".", "+")+".pt").unlink()
        
    def save_checkpoint(self,  model):
        if self.save: torch.save(model.state_dict(), str(self.path))

    
class EarlyStoppingL(ES):
    def __init__(self, patience, save, path):
        super().__init__(patience, save, path)
        self.val_loss_min = np.Inf
            
    def __call__(self, val_loss, val_acc, model):
    #todo loss/acc option
        if val_loss >= self.val_loss_min:
            self.counter += 1
            if self.counter >= self.patience:
                return True
        else:
            self.val_loss_min = val_loss
            self.counter = 0
            self.save_checkpoint(model)
        return False
    

class EarlyStoppingA(ES):
    def __init__(self, patience, save, path):
        super().__init__(patience, save, path)
        self.val_acc_max = -1.0
    
    def __call__(self, val_loss, val_acc, model):
    #todo loss/acc option
        if isinstance(val_acc, dict):
            val_acc = val_acc["f1_weighted"]
        if val_acc <= self.val_acc_max:
            self.counter += 1
            if self.counter >= self.patience:
                return True
        else:
            self.val_acc_max = val_acc
            self.counter = 0
            self.save_checkpoint(model)
        return False