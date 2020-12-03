import models
import solver
import dataloader

import torch
from torch.utils.data import DataLoader
import torchvision
from torchvision import datasets
import mlflow
import hydra
import omegaconf
import numpy as np

from logging import getLogger
from pathlib import Path
from collections import defaultdict
import random
import os


SEED = 1234
random.seed(SEED)
np.random.seed(SEED)
os.environ["PYTHONHASHSEED"] = str(SEED)
torch.manual_seed(SEED)


def log_params(dic, name=None):
    for key, values in dic.items():
        if type(values) == omegaconf.dictconfig.DictConfig:
            if name is not None:
                key = name + "." + key
            log_params(values, key)
        else:
            if name is not None:
                key = name + "." + key
            mlflow.log_param(key, values)


@hydra.main("config.yaml")
def main(cfg):
    logger = getLogger(__name__)

    mlflow.set_tracking_uri('file://' + hydra.utils.get_original_cwd() + '/mlruns')
    mlflow.set_experiment(cfg.ex_name)

    with mlflow.start_run() as run:
        log_params(cfg)

        net = models.get_model(cfg.net.name, **cfg.net.kwargs)
        logger.info(net)

        transforms = torchvision.transforms.Compose([
            torchvision.transforms.ToTensor()
            ])
        # train = datasets.MNIST("../data", train=True, download=True, transform=transforms)
        # test = datasets.MNIST("../data", train=False, transform=transforms)

        train = dataloader.MyMNIST(root="../data", train=True, download=True, transform=transforms,
                limit_data=[0,cfg.num_data])

        val = dataloader.MyMNIST(root="../data", train=True, download=True, transform=transforms,
                limit_data=[cfg.num_data, -1])

        test = dataloader.MyMNIST(root="../data", train=False, download=True, transform=transforms,)

        train_dataloader = DataLoader(train, **cfg.dataloader.kwargs,)
        val_dataloader = DataLoader(train, **cfg.dataloader.kwargs,)
        test_dataloader = DataLoader(test, **cfg.dataloader.kwargs,)

        optimizer = getattr(torch.optim, cfg.optim.name)(net.parameters(), **cfg.optim.kwargs)
        scheduler = getattr(torch.optim.lr_scheduler, cfg.scheduler.name)(optimizer, **cfg.scheduler.kwargs)

        loss = getattr(torch.nn, cfg.loss)()

        trainer= solver.Solver(net, train_dataloader, val_dataloader, test_dataloader,
                optimizer, scheduler, lossfunc=loss, **cfg.solver.kwargs)

        res = trainer.train()
        for key, item in res.items():
            mlflow.log_metric(key, item)

        res = trainer.test()
        mlflow.log_metric("loss", res["loss"])
        mlflow.log_metric("acc", res["acc"])
        trainer.save()

if __name__ == "__main__":
    main()
