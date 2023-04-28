#!/usr/bin/env python
# coding: utf-8

from pathlib import Path
import yaml
import numpy as np
from tqdm import tqdm
import torch
from torch import multiprocessing
import sys

from data import get_dataset
from models import get_model
from optimizer import get_optim
from training_methods import get_train


def find_pars_yaml(argv = []):
    paths = []
    ret = []
    if len(argv) >= 2:
        if(argv[1] != "glob"):
            for x in range(1, len(argv)):
                paths.append(Path("../experiments/" + argv[x]))
        else:
            for x in range(2, len(argv)):
                paths.extend(Path("../experiments/").glob(argv[x]))
    else:
        paths = Path("../experiments/").glob("*.yml")
    for p in paths:
        yml = yaml.safe_load(p.open())
        if yml.get("alldone"):
            continue
        else:
            ret.append(p)
    return ret

def num2conf(num, lens):
    left = num
    res = [0]*len(lens)
    for ix in range(len(lens)-1, -1, -1):
        res[ix] = left % lens[ix]
        left = int(left/lens[ix])
    return res

def dict2dev(d, dev):
    return {x : d[x].to(dev) for x in d}

def train1setting(settings, save_path):
    pr_nr = int(multiprocessing.current_process().name.split("-")[1])
    d = {0:0, 1:1, 2:0, 3:1, 4:2}
    device = torch.device("cuda:"+str(d[pr_nr%2]) if torch.cuda.is_available() else "cpu")
    #device = torch.device("cuda:3")
    save_path.mkdir(parents=True, exist_ok=True)
    ds = get_dataset(settings, device)
    yaml.dump(settings, (save_path/"settings.yml").open(mode="w"))
    if type(settings["statrep"])==int:
        start = 0
        end = settings["statrep"]
    else:
        start = int(settings["statrep"].split("_")[0])
        end = int(settings["statrep"].split("_")[1])

    for rep in range(start, end):
        p = save_path / ("stats_"+str(rep)+".pkl")
        if p.exists():
            continue
        #print("rep", rep)
        ds.setRep(rep)
        #torch.cuda.empty_cache()
        torch.manual_seed(rep)
        model = get_model(settings, ds.num_node_features, ds.num_classes).to(device)
        opt = get_optim(model, settings)
        train = get_train(settings)
        model_path = save_path / ("model_"+str(rep)+".pt")
        stats = train(ds, model, opt, settings, save_path = model_path)
        if settings["save_models"] and settings["early_stopping"]==False:
            torch.save(model.state_dict(), model_path)
        stats.to_pickle(p)

#glue method to use imap
def f(args):
    i, lens, keys, settings_dict, save_path = args
    conf = num2conf(i, lens)
    current_dict = {keys[i]:settings_dict[keys[i]][conf[i]] for i in range(len(conf))}
    train1setting(current_dict, save_path/("setting_"+str(i)))
        
def run1exp(save_path, settings_dict, p):
    setting_ls = list(settings_dict.items())
    lens = [len(x[1]) for x in setting_ls]
    keys = [x[0] for x in setting_ls]
    n = np.prod(np.array(lens))
    it = list(zip(range(n), [lens]*n, [keys]*n, [settings_dict]*n, [save_path]*n))
    tmp = list(tqdm(p.imap(f, it), total=len(it)))


def main(argv):
    paths2do = find_pars_yaml(argv)
    print("#"*30)
    print("will run "+str(len(paths2do))+" experiments:")
    print(paths2do)
    print("#"*30)
    multiprocessing.set_start_method("forkserver")
    with multiprocessing.Pool(2) as p:
        for count, path in enumerate(paths2do):
            print((count+1), "of", len(paths2do), "doing", path)
            yml = yaml.safe_load(path.open())
            print(yml)
            print("#"*30)
            save_path = Path("../results/"+path.parts[-1].split(".")[0]+"/")
            run1exp(save_path, yml, p)
            yml["alldone"] = True
            yaml.dump(yml, path.open(mode="w"))


if __name__ == "__main__":
    main(sys.argv)
