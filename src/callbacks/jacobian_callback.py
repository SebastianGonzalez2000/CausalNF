from lightning.pytorch.callbacks import Callback

from src.utils.jacobian_utils import jacobian_dynamics

class JacobianCallback(Callback):
    def on_validation_epoch_end(self, trainer, pl_module):
        jac = jacobian_dynamics(pl_module.model, trainer.datamodule.predict_dataloader())

        for row in range(jac.size()[0]):
            for col in range(jac.size()[1]):
                pl_module.log(f'Jacobian[{row}][{col}]', jac[row][col])