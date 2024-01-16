from pytorch_lightning.callbacks import Callback

from src.utils.jacobian_utils import jacobian_dynamics

class JacobianCallback(Callback):
    def on_train_epoch_end(self, trainer, pl_module):
        jac = jacobian_dynamics(pl_module, trainer.datamodule.predict_dataloader())
        pl_module.jac = jac
        pl_module.jac_l1 = jac.norm(p=1)
        pl_module.log("Jacobian l1", pl_module.jac_l1)
        for row in range(jac.size()[0]):
            for col in range(row):
                pl_module.log(f'Jacobian[{row}][{col}]', jac[row][col])