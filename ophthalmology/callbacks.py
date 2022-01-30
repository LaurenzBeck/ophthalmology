# -*- coding: utf-8 -*-
"""Pytorch Lightning Callbacks"""

import warnings
from functools import partial
from itertools import cycle
from typing import List, Optional, Sequence, Tuple, Union

import mlflow
import optuna
import pytorch_lightning as pl
import snoop
import torch
import torchmetrics
from loguru import logger as log
from matplotlib import pyplot as plt
from torch.nn import functional as F
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

from ophthalmology import visualization

with optuna._imports.try_import() as _imports:
    from pytorch_lightning import LightningModule, Trainer
    from pytorch_lightning.callbacks import Callback

if not _imports.is_successful():
    Callback = object  # type: ignore # NOQA
    LightningModule = object  # type: ignore # NOQA
    Trainer = object  # type: ignore # NOQA


class OptunaPruningCallback(Callback):
    """Optuna callback to prune unpromising trials.

    See `the example <https://github.com/optuna/optuna-examples/blob/
    main/pytorch/pytorch_lightning_simple.py>`__
    if you want to add a pruning callback which observes accuracy.

    This version also tags the current mlflow run with pruned:'pruned'

    Args:
        trial:
            A :class:`~optuna.trial.Trial` corresponding to the current evaluation of the
            objective function.
        monitor:
            An evaluation metric for pruning, e.g., ``val_loss`` or
            ``val_acc``. The metrics are obtained from the returned dictionaries from e.g.
            ``pytorch_lightning.LightningModule.training_step`` or
            ``pytorch_lightning.LightningModule.validation_epoch_end`` and the names thus depend on
            how this dictionary is formatted.
    """

    def __init__(self, trial: optuna.trial.Trial, monitor: str) -> None:
        _imports.check()
        super().__init__()

        self._trial = trial
        self.monitor = monitor

    def on_validation_end(
        self, trainer: Trainer, pl_module: LightningModule
    ) -> None:
        epoch = pl_module.current_epoch

        current_score = trainer.callback_metrics.get(self.monitor)
        if current_score is None:
            message = (
                "The metric '{}' is not in the evaluation logs for pruning. "
                "Please make sure you set the correct metric name.".format(
                    self.monitor
                )
            )
            warnings.warn(message)
            return

        self._trial.report(current_score, step=epoch)
        if self._trial.should_prune():
            mlflow.set_tag("optuna", "pruned")
            message = "Trial was pruned at epoch {}.".format(epoch)
            raise optuna.TrialPruned(message)


class LogDataSamplesCallback(pl.Callback):
    def __init__(self, dataset: Dataset, rows: int = 4):
        """Callback to visualize the train dataset with matplotlib
        and log the results in mlflow in the current active run.

        Args:
            dataset (Dataset): pytorch Dataset (will be directly indexed without a Dataloader.)
            rows (int, optional): How many samples will be in one row. Total number of samples will be rows^2. Defaults to 5.
        """
        super(LogDataSamplesCallback, self).__init__()
        self.dataset = dataset
        self.rows = rows

    def on_fit_start(
        self, trainer: "pl.Trainer", pl_module: "pl.LightningModule"
    ) -> None:
        fig = visualization.visualize_samples_from_dataset(
            self.dataset, self.rows
        )
        mlflow.log_figure(fig, "train_samples.png")
        plt.close(fig)

        return super().on_fit_start(trainer, pl_module)


class LogSignalPropagationPlotCallback(pl.Callback):
    def __init__(self, input_shape: List[int] = [64, 3, 256, 256]):
        """Callback to visualize the signal propagation at initialization with matplotlib
        and log the results in mlflow in the current active run.
        See: https://github.com/mehdidc/signal_propagation_plot/blob/main/signal_propagation_plot/pytorch.py

        Args:
            input_shape (List[int], optional): Input Size of the model.
        """
        super().__init__()
        self.input_shape = input_shape

    def on_fit_start(
        self, trainer: "pl.Trainer", pl_module: "pl.LightningModule"
    ) -> None:
        x = torch.randn(*self.input_shape).to(pl_module.device)

        name_values_squared_mean = LogSignalPropagationPlotCallback.get_average_channel_squared_mean_by_depth(
            pl_module.model, x
        )
        fig = visualization.visualize_signal_propagation(
            name_values_squared_mean, title="mean channel squared mean"
        )
        mlflow.log_figure(fig, "mean_channel_squared_mean.png")
        plt.close(fig)

        name_values_variance = LogSignalPropagationPlotCallback.get_average_channel_variance_by_depth(
            pl_module.model, x
        )
        fig = visualization.visualize_signal_propagation(
            name_values_variance, title="mean channel variance"
        )
        mlflow.log_figure(fig, "mean_channel_variance.png")
        plt.close(fig)

        return super().on_fit_start(trainer, pl_module)

    @staticmethod
    def get_average_channel_squared_mean_by_depth(model, *args, **kwargs):
        acts = LogSignalPropagationPlotCallback.extract_activations(
            model, *args, **kwargs
        )
        values = []
        for name, tensor in acts:
            values.append(
                (
                    name,
                    LogSignalPropagationPlotCallback.average_channel_squared_mean(
                        tensor
                    ),
                )
            )
        return values

    @staticmethod
    def get_average_channel_variance_by_depth(model, *args, **kwargs):
        acts = LogSignalPropagationPlotCallback.extract_activations(
            model, *args, **kwargs
        )
        values = []
        for name, tensor in acts:
            values.append(
                (
                    name,
                    LogSignalPropagationPlotCallback.average_channel_variance(
                        tensor
                    ),
                )
            )
        return values

    @staticmethod
    def average_channel_squared_mean(x):
        if x.ndim == 4:
            return (x.mean(dim=(0, 2, 3)) ** 2).mean().item()
        elif x.ndim == 2:
            return (x ** 2).mean().item()
        else:
            return -1.0  #! Just for debugging involution Layer
            raise ValueError(f"not supported shape: {x.shape}")

    @staticmethod
    def average_channel_variance(x):
        if x.ndim == 4 and x.shape[2] > 1 and x.shape[3] > 1:
            return x.var(dim=(0, 2, 3)).mean().item()
        elif x.ndim == 2 or (
            x.ndim == 4 and x.shape[2] == 1 and x.shape[3] == 1
        ):
            return x.var(dim=0).mean().item()
        else:
            return -1.0  #! Just for debugging involution Layer
            raise ValueError(f"not supported shape: {x.shape}")

    @staticmethod
    def extract_activations(model, *args, **kwargs):
        acts = []
        handles = []
        for (
            n,
            m,
        ) in (
            model.named_children()
        ):  # switch to named_modules and remove next line for full depth
            for name, module in m.named_children():
                handle = module.register_forward_hook(
                    partial(
                        LogSignalPropagationPlotCallback.hook,
                        name=(".".join([n, name])),
                        store=acts,
                    )
                )
                handles.append(handle)
        model(*args, **kwargs)
        for handle in handles:
            handle.remove()
        return acts

    def hook(self, input, output, store=None, name=None):
        if store is None:
            store = []
        store.append((name, output))


class SSLOnlineEvaluator(pl.Callback):  # pragma: no cover
    """Attaches an MLP for fine-tuning using the standard self-supervised protocol.
    Code adapted from: https://github.com/PyTorchLightning/Lightning-Bolts/blob/master/pl_bolts/callbacks/ssl_online.py#L11-L139

    Example::
        # your model must have 2 attributes
        model = Model()
        model.z_dim = ... # the representation dim
        model.num_classes = ... # the num of classes in the model
        online_eval = SSLOnlineEvaluator(
            z_dim=model.z_dim,
            num_classes=model.num_classes,
            dataset='imagenet'
        )
    """

    def __init__(
        self,
        data_loader: DataLoader,
        drop_p: float = 0.2,
        hidden_dim: Optional[int] = None,
        z_dim: int = None,
        num_classes: int = None,
    ):
        """
        Args:
            dataset: if stl10, need to get the labeled batch
            drop_p: Dropout probability
            hidden_dim: Hidden dimension for the fine-tune MLP
            z_dim: Representation dimension
            num_classes: Number of classes
        """
        super().__init__()

        self.hidden_dim = hidden_dim
        self.drop_p = drop_p
        self.z_dim = z_dim
        self.num_classes = num_classes
        self.data_loader = data_loader
        self.data_flow = cycle(data_loader)

    def on_pretrain_routine_start(
        self, trainer: Trainer, pl_module: LightningModule
    ) -> None:
        from pl_bolts.models.self_supervised.evaluator import SSLEvaluator

        pl_module.non_linear_evaluator = SSLEvaluator(
            n_input=self.z_dim,
            n_classes=self.num_classes,
            p=self.drop_p,
            n_hidden=self.hidden_dim,
        ).to(pl_module.device)

        self.optimizer = torch.optim.Adam(
            pl_module.non_linear_evaluator.parameters(), lr=1e-4
        )

        self.confusion_matrix = torchmetrics.ConfusionMatrix(num_classes=5).to(
            pl_module.device
        )

    def to_device(
        self, batch: Sequence, device: Union[str, torch.device]
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        x, y = batch

        x = x.to(device)
        y = y.to(device)

        return x, y

    def on_train_batch_end(
        self,
        trainer_: Trainer,
        pl_module: LightningModule,
        outputs_: Sequence,
        batch_: Sequence,
        batch_idx_: int,
        # dataloader_idx_: int,
    ) -> None:
        batch = next(self.data_flow)
        x, y = self.to_device(batch, pl_module.device)

        with torch.no_grad():
            representations = pl_module(x)

        # forward pass
        mlp_logits = pl_module.non_linear_evaluator(representations)  # type: ignore[operator]
        mlp_loss = F.cross_entropy(mlp_logits, y)

        pl_module.log("online/loss", mlp_loss, on_step=True, on_epoch=False)

        # update finetune weights
        mlp_loss.backward()
        self.optimizer.step()
        self.optimizer.zero_grad()

        # log metrics
        acc = torchmetrics.functional.accuracy(
            mlp_logits.softmax(-1), y, num_classes=5
        )
        f1 = torchmetrics.functional.f1(
            mlp_logits.softmax(-1), y, num_classes=5
        )
        pl_module.log(
            "online/acc", acc, on_step=True, on_epoch=False, prog_bar=True
        )
        pl_module.log("online/F1", f1, on_step=True, on_epoch=False)

    @snoop()
    def on_validation_epoch_end(
        self, trainer: "pl.Trainer", pl_module: "pl.LightningModule"
    ) -> None:
        """Do One Epoch of finetuning on the head and compute confusion matrix."""
        log.debug("Doing one Epoch of evaluation on the labeled test set:")
        with torch.enable_grad():
            for x, y in tqdm(self.data_loader):
                batch = next(self.data_flow)
                x, y = self.to_device(batch, pl_module.device)

                with torch.no_grad():
                    representations = pl_module(x)

                # forward pass
                mlp_logits = pl_module.non_linear_evaluator(representations)  # type: ignore[operator]
                mlp_loss = F.cross_entropy(mlp_logits, y)

                pl_module.log("online/loss", mlp_loss)

                # update finetune weights
                mlp_loss.backward()
                self.optimizer.step()
                self.optimizer.zero_grad()

                # log metrics
                self.confusion_matrix(mlp_logits.softmax(-1), y)
                acc = torchmetrics.functional.accuracy(
                    mlp_logits.softmax(-1), y, num_classes=5
                )
                f1 = torchmetrics.functional.f1(
                    mlp_logits.softmax(-1), y, num_classes=5
                )
                pl_module.log("online/acc", acc)
                pl_module.log("online/F1", f1)

        confusion_matrix = self.confusion_matrix.compute()
        fig = visualization.plot_confusion_matrix(
            confusion_matrix,
            classes=["circle", "crescent", "double-crescent", "other", "spot"],
            normalize=False,
        )
        mlflow.log_figure(fig, "val_confusion_matrix.png")
        plt.close(fig)
        self.confusion_matrix.reset()

        return super().on_validation_epoch_end(trainer, pl_module)
