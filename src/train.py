import torch
from torch import nn
import lightning.pytorch as pl

import hydra

from src.utils.jacobian_utils import jacobian_dynamics

@hydra.main(version_base="1.3", config_path="../configs", config_name="train.yaml")
def main(cfg):
    print(cfg)

    datamodule = hydra.utils.instantiate(cfg.data)
    module = hydra.utils.instantiate(cfg.module)
    trainer = hydra.utils.instantiate(cfg.trainer)

    trainer.fit(module, datamodule=datamodule)
    trainer.test(model=module, datamodule=datamodule)

    jac = jacobian_dynamics(module.model, datamodule.train_dataloader())
    print(jac)

if __name__ == "__main__":
    main()