import torch
from torch import nn
import torch.nn.functional as F


#gsam helper functions, adapted from torch.utils model2vec and vec2model
def grads_to_vec(parameters):
    vec = []
    for param in parameters:
        vec.append(param.grad.view(-1))
    return torch.cat(vec)

# def grads_to_vec(model):
#     vec = []
#     print("in g2v")
#     for name, param in model.named_parameters():
#         print(name, param)
#         print(param.shape)
#         print(param.grad)
#         vec.append(param.grad.view(-1))
#     print("all appended")
#     return torch.cat(vec)




def vec_to_grads(vec, parameters):
    # Pointer for slicing the vector for each parameter
    pointer = 0
    for param in parameters:
        # The length of the parameter
        num_param = param.numel()
        # Slice the vector, reshape it, and replace the old data of the parameter
        param.grad = vec[pointer:pointer + num_param].view_as(param).data

        # Increment the pointer
        pointer += num_param
        
def projAonB(a,b):
    bn = F.normalize(b, p=2, dim=0)
    return torch.dot(a,bn)*bn
        


        
def disable_running_stats(model):
    def _disable(module):
        if isinstance(module, nn.BatchNorm1d):
            module.backup_momentum = module.momentum
            module.momentum = 0
    model.apply(_disable)

def enable_running_stats(model):
    def _enable(module):
        if isinstance(module, nn.BatchNorm1d) and hasattr(module, "backup_momentum"):
            module.momentum = module.backup_momentum
    model.apply(_enable)