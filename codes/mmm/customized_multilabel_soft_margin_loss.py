import torch
import torch.nn as nn
from torch._overrides import has_torch_function, handle_torch_function


logsigmoid = torch._C._nn.log_sigmoid


def _multilabel_soft_margin_loss(input, target, weight=None, size_average=None,
                                 reduce=None, reduction='mean'):
    if not torch.jit.is_scripting():
        tens_ops = (input, target)

        # 基本ここでTrueになることはないが，なった場合エラーを吐く可能性が高い
        if (
            any([type(t) is not torch.Tensor for t in tens_ops])
            and has_torch_function(tens_ops)
        ):
            return handle_torch_function(
                _multilabel_soft_margin_loss, tens_ops, input, target,
                weight=weight, size_average=size_average,
                reduce=reduce, reduction=reduction
            )

    if size_average is not None or reduce is not None:
        reduction = nn._reduction.legacy_get_string(size_average, reduce)

    loss = -(target * logsigmoid(input) + (1 - target) * logsigmoid(-input))

    if weight is not None:
        loss = loss * weight

    loss = loss.sum(dim=1) / input.size(1)  # only return N loss values

    if reduction == 'none':
        ret = loss
    elif reduction == 'mean':
        ret = loss.mean()
    elif reduction == 'sum':
        ret = loss.sum()
    else:
        ret = input
        raise ValueError(reduction + " is not valid")
    return ret


class _Loss(nn.Module):
    def __init__(self, size_average=None, reduce=None, reduction='mean'):
        super(_Loss, self).__init__()
        if size_average is not None or reduce is not None:
            self.reduction = nn._reduction.legacy_get_string(
                size_average, reduce
            )
        else:
            self.reduction = reduction


class _WeightedLoss(_Loss):
    def __init__(self, weight=None, size_average=None,
                 reduce=None, reduction='mean'):
        super(_WeightedLoss, self).__init__(size_average, reduce, reduction)
        self.register_buffer('weight', weight)


class CustomizedMultiLabelSoftMarginLoss(_WeightedLoss):
    '''
    誤差伝播の重みを各データ毎に指定できるようMultiLabelSoftMarginLossを修正
    '''
    def __init__(self, weight=None, size_average=None,
                 reduce=None, reduction='mean'):
        super(CustomizedMultiLabelSoftMarginLoss, self).__init__(
            weight, size_average, reduce, reduction)

    def forward(self, input, target, weight=None):
        return _multilabel_soft_margin_loss(
            input, target, weight=weight, reduction=self.reduction
        )
