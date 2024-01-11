import torch
from torch import nn
import lightning.pytorch as pl

import hydra
import os
from src.utils.jacobian_utils import jacobian_dynamics
from src.callbacks.jacobian_callback import JacobianCallback

@hydra.main(version_base="1.3", config_path="../configs", config_name="train.yaml")
def main(cfg):

    datamodule = hydra.utils.instantiate(cfg.data)
    module = hydra.utils.instantiate(cfg.module)
    tb_logger = pl.loggers.TensorBoardLogger(save_dir="tb_logs")
    trainer = hydra.utils.instantiate(cfg.trainer, callbacks=[JacobianCallback()],logger=tb_logger)
    #trainer = hydra.utils.instantiate(cfg.trainer, logger=tb_logger)
    

    trainer.fit(module, datamodule=datamodule)
    trainer.test(model=module, datamodule=datamodule)

    jac = jacobian_dynamics(module.model, datamodule.predict_dataloader())
    print(jac)

if __name__ == "__main__":
    main()