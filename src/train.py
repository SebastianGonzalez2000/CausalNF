import torch
from torch import nn
import pytorch_lightning as pl

import hydra
import os
from src.utils.jacobian_utils import jacobian_dynamics
from src.callbacks.jacobian_callback import JacobianCallback

@hydra.main(version_base="1.3", config_path="../configs", config_name="train.yaml")
def main(cfg): 

    print(cfg)

    logdir = "checkpoints/" + cfg.data.name + "/" + cfg.module.name 
    if not os.path.exists(logdir):
        os.makedirs(logdir)     

    wandb_logger = pl.loggers.WandbLogger(save_dir = logdir, project = "causalnf")
    datamodule = hydra.utils.instantiate(cfg.data)
    module = hydra.utils.instantiate(cfg.module)
    #tb_logger = pl.loggers.WandbLogger(save_dir="tb_logs")
    trainer = hydra.utils.instantiate(cfg.trainer, callbacks=[JacobianCallback()],logger=wandb_logger)
    #trainer = hydra.utils.instantiate(cfg.trainer, logger=tb_logger)
    

    trainer.fit(module, datamodule=datamodule)

    # jac = jacobian_dynamics(module.model, datamodule.predict_dataloader())
    # print(jac)

if __name__ == "__main__":
    main()