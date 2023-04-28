from torch import optim
from sam import SAM

def get_optim(model, settings):
    optim_d = {"SGD": optim.SGD,
               "RMSprop": optim.RMSprop,
               "Adam": optim.Adam,
               "NAdam": optim.NAdam,
               "AdamW": optim.AdamW
              }
    base_settings = {x[0].split(":")[1]:settings[x[0]] for x in settings.items() if (len(x[0].split(":"))==2) and (x[0].split(":")[0]=="optim")}
    
    opt_ls = settings["base_optim"].split("_")
    base_opt = opt_ls[0]
    if len(opt_ls)==2:
        base_settings["momentum"]=float(opt_ls[1])   #to do SGD_0.9
    
    #params = model.parameters()
    params = model.create_wd_groups(settings)
    
    if settings["sam"] == "nosam":
        return optim_d[base_opt](params, **base_settings)
    else:
        ls = settings["sam"].split("_")
        adapt = ls[0]=="asam"
        return SAM(params, optim_d[base_opt], **base_settings, rho=float(ls[1]), adaptive = adapt)