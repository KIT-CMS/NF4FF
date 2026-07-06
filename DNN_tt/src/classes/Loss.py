from abc import ABC, abstractmethod
from functools import partial
from itertools import product
from typing import Any, Callable, Dict, List, Literal, Tuple, Union

import opt_einsum as oe
import torch as t
import torch.nn.functional as F
from torch.nn.modules.loss import _Loss
import torch.nn as nn
import classes.helper as helper
#from CODE.LOSS.LikelihoodsAndUncertainties import (Likelihood, UncertaintyObjects)
#from CODE.TAYLORANALYSIS import TCExtraction

# ---------------------

einsum = partial(oe.contract, backend="torch", optimize="auto")



def _BCE_loss(
    input: t.Tensor,
    target: t.Tensor,
    weights_class: t.Tensor,
    accumulation_function: Callable = t.mean,
) -> t.Tensor:
    if target.dim() == 1:
        return t.nn.BCELoss(
            weights_class.squeeze(),
            reduction=accumulation_function.__name__,
        )(input.squeeze(), target.squeeze())

    class_like = t.clip(input, 1e-9, 1 - 1e-9).log() * target
    not_class_like = t.clip(1 - input, 1e-9, 1 - 1e-9).log() * (1 - target)
    return accumulation_function(
        einsum(
            "i..., i -> i...",
            -(class_like + not_class_like),
            weights_class,
        )
    )


def _CE_loss(
    input: t.Tensor,
    target: t.Tensor,
    weights_class: t.Tensor,
    accumulation_function: Callable = t.mean,
) -> t.Tensor:
    if target.dim() == 1:
        target = t.stack([(~target.to(t.bool)).to(t.float), target.to(t.float)]).T
    input = t.clip(input, 1e-9, 1 - 1e-9)
    return accumulation_function(-(target * einsum("i, ij -> ij", weights_class, input.log())).sum(axis=1))







def _BCE_weight_normalized(
    input: t.Tensor,
    target: t.Tensor,
    weights_class: t.Tensor,
    accumulation_function: Callable = t.mean,
) -> t.Tensor:
    input = input.clip(1e-6, 1.0 - 1e-6)
    if target.dim() == 1:
        loss = -(target * input.log() + (1 - target) * (1 - input).log())
        loss = loss * weights_class.squeeze()
    else:
        class_like = input.log() * target
        not_class_like = (1 - input).log() * (1 - target)
        loss = -(class_like + not_class_like).sum(dim=-1) * weights_class.squeeze()

    if accumulation_function.__name__ == "mean":
        return loss.sum() / weights_class.abs().sum().clamp(min=1e-6)
    return loss.sum()





class PretrainingLoss(_Loss):
    def __init__(
        self,
        device: str,
        signal_category: Union[int, list[int], tuple[int]] = 1,
        background_category: Union[int, list[int], tuple[int]] = 0,
        n_classes: int = 1,
        final_activation: str = "Sigmoid",
        size_average: Any = None,
        reduce: Any = None,
        reduction: str = "mean",
        accumulation_function: Callable = t.mean,
        **kwargs: Any,
    ) -> None:
        super().__init__(size_average, reduce, reduction)

        self.mode = "CE"

        self.device = device

        self.signal_category = signal_category
        self.multiclass = n_classes > 1
        self.one_class_multiclass = helper.is_one_class_multiclass(n_classes, signal_category, background_category)

        self.final_activation = final_activation
        self.is_sigmoid = final_activation.lower() == "sigmoid"

        self.accumulation_function = accumulation_function

        self.tracked_unscaled_loss: float = float("inf")
        self.tracked_unscaled_partial_losses = []

        self._base_BCE_loss = _BCE_loss
        self._base_CE_loss = _CE_loss

    def to(self, device: str) -> "PretrainingLoss":
        self.device = device
        return self

    def track_unscaled_loss(self, value: t.Tensor) -> t.Tensor:
        if not value.dim():
            self.tracked_unscaled_loss = value.detach().cpu().item()
        else:
            self.tracked_unscaled_loss = value.sum().detach().cpu().item()
        return value

    def forward(
        self,
        input: t.Tensor,
        target: t.Tensor,
        weights_class: t.Tensor,
        **kwargs: Any,
    ) -> t.Tensor:

        target = target.to(input.device)
        weights_class = weights_class.to(input.device)

        if not self.multiclass and not self.one_class_multiclass:
            return self.track_unscaled_loss(
                self._base_BCE_loss(
                    input=input,
                    target=target,
                    weights_class=weights_class,
                    accumulation_function=self.accumulation_function,
                ),
            )

        loss_function = self._base_BCE_loss if self.is_sigmoid else self._base_CE_loss

        if not self.one_class_multiclass:
            return self.track_unscaled_loss(
                loss_function(
                    input=input,
                    target=target,
                    weights_class=weights_class,
                    accumulation_function=self.accumulation_function,
                ),
            )

        target = target[:, self.signal_category]

        if target.dim() == 2:
            target = target.sum(axis=1)  # type: ignore

        return self.track_unscaled_loss(
            loss_function(
                input=input,
                target=target,
                weights_class=weights_class,
                accumulation_function=self.accumulation_function,
            ),
        )


class PretrainingBCELossWeightNormalized(nn.Module):
    def __init__(self, eps: float = 1e-6):
        super().__init__()
        self.eps = eps

        # tracking
        self.tracked_total_loss = float("inf")
        self.tracked_bce_loss = float("inf")

    def track(self, loss: t.Tensor):
        self.tracked_total_loss = loss.detach().cpu().item()
        self.tracked_bce_loss = self.tracked_total_loss
        return loss

    def forward(
        self,
        input: t.Tensor,
        target: t.Tensor,
        weights_class: t.Tensor,
    ) -> t.Tensor:

        input = input.clamp(self.eps, 1.0 - self.eps)

        # BCE (manual, like your style)
        bce = -(
            target * input.log()
            + (1 - target) * (1 - input).log()
        )

        bce = bce * weights_class.squeeze()

        loss = (
            bce.sum()
            / weights_class.abs().sum().clamp(min=self.eps)
        )

        return self.track(loss)

class PretrainingSqueezedLossWeightNormalized(PretrainingLoss):
    def __init__(
        self,
        squeeze_lambda: float = 0.000001,  # 0.0001
        penalty_lower_bound: float = -1000,  # -0.15
        penalty_upper_bound: float = 2.2,  # 0.15
        dynamic_penalty_scaling: bool = False,
        epoch_decay: bool = False,
        *args: Any,
        **kwargs: Any,
    ):
        super().__init__(*args, **kwargs)
        self.squeeze_lambda = squeeze_lambda
        self.penalty_lower_bound = penalty_lower_bound
        self.penalty_upper_bound = penalty_upper_bound
        self._base_BCE_loss = _BCE_weight_normalized
        self.dynamic_penalty_scaling = dynamic_penalty_scaling
        self.epoch_decay = epoch_decay

        # Tracking fields consumed by train_dnn_squeezed_loss.
        self.tracked_base_loss = float("inf")
        self.tracked_penalty_loss = float("inf")

    def forward(
        self,
        input: t.Tensor,
        target: t.Tensor,
        weights_class: t.Tensor,
        **kwargs: Any,
    ) -> t.Tensor:
        target = target.to(input.device)
        weights_class = weights_class.to(input.device)


        loss_function = self._base_BCE_loss

        base_loss = loss_function(
            input=input,
            target=target,
            weights_class=weights_class,
            accumulation_function=self.accumulation_function,
        )

        clamped_input = input.clip(1e-6, 1.0 - 1e-6)
        logits = t.log(clamped_input / (1.0 - clamped_input))

        high = F.relu(logits - self.penalty_upper_bound)
        low = F.relu(self.penalty_lower_bound - logits)

        penalty_unreduced = ((high ** 2) + (low ** 2)).sum(dim=-1) * weights_class.squeeze()

        penalty_loss = penalty_unreduced.sum()


        if self.dynamic_penalty_scaling:
            factor = (penalty_loss.detach() / (base_loss.detach() + 1e-8)) * 1e-5
        else:
            factor = self.squeeze_lambda
            
        # Expose scalar components for logging/debug panels.
        self.tracked_base_loss = base_loss.detach().cpu().item()
        self.tracked_penalty_loss = (factor * penalty_loss).detach().cpu().item()

        return self.track_unscaled_loss(base_loss + (factor * penalty_loss))
