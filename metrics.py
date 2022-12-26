import torch


def precision(y_true, y_pred, reduce_axes, epsilon=1e-6):
    """
    Differentiable precision score with real-valued (0 to 1 inclusive) probability input

    precision = TP / (TP + FP)

    Parameters
    ----------
    y_true : torch.Tensor
        Values must be between 0 and 1.
    y_pred : torch.Tensor
        Values must be between 0 and 1.
    reduce_axes : tuple of int
        Axes to sum over for numerator and denominator.
    epsilon : float
        Minimum denominator value to prevent division by zero.

    Returns
    -------
    score : torch.Tensor
        Precision score tensor with the unreduced axes.
    """
    numerator = (y_pred * y_true).sum(reduce_axes)
    denominator = y_pred.sum(reduce_axes).clamp(min=epsilon)
    score = numerator / denominator
    return score


def recall(y_true, y_pred, reduce_axes, epsilon=1e-6):
    """
    Differentiable recall score with real-valued probability input

    Note: recall is the same as precision with flipped y_true and y_pred
    """
    return precision(y_true=y_pred, y_pred=y_true, reduce_axes=reduce_axes, epsilon=epsilon)


def f_beta(y_true, y_pred, reduce_axes, beta=1., epsilon=1e-6):
    """
    Differentiable F-beta score

    F_beta = (1 + beta^2) * TP / [beta^2 * (TP + FN) + (TP + FP)]
    = (1 + beta^2) (true * pred) / [beta^2 * true + pred]

    Using this formula, we don't have to use precision and recall.

    F1 score is an example of F-beta score with beta=1.
    """
    beta2 = beta ** 2   # beta squared
    numerator = (1 + beta2) * (y_true * y_pred).sum(reduce_axes)
    denominator = beta2 * y_true.sum(reduce_axes) + y_pred.sum(reduce_axes)
    denominator = denominator.clamp(min=epsilon)
    return numerator / denominator


def dice(y_true, y_pred, reduce_axes, beta=1., epsilon=1e-6):
    """
    Compute soft dice coefficient according to V-Net paper.
    (https://arxiv.org/abs/1606.04797)

    Unlike F-beta score, dice uses squared probabilities
    instead of probabilities themselves for denominator.
    Due to the squared entries, gradients will be y_true and y_pred instead of 1.
    """
    beta2 = beta ** 2   # beta squared
    numerator = (1 + beta2) * (y_true * y_pred).sum(reduce_axes)
    denominator = beta2 * y_true.square().sum(reduce_axes) + y_pred.square().sum(reduce_axes)
    denominator = denominator.clamp(min=epsilon)
    return numerator / denominator


def confusion_matrix_loss(y_true, y_pred, metric="dice", average="samples", beta=1, epsilon=1e-6):
    """
    Compute a loss function using confusion matrix.
    Supports precision, recall, F-beta score (including F-1 score), and dice coefficient.

    Parameters
    ----------
    y_true : torch.Tensor
        Has shape [N, C, ...], where N is batch size and C is channel size.
        The following dimensions are spatial, such as depth, height, and width.

    y_pred : torch.Tensor
        Has shape [N, C, ...], the same as y_true.

    metric : str
        One of precision, recall, f_beta, dice.

    average : str
        One of micro, samples, macro

    beta : float
        Beta coefficient for F-beta and dice.
        beta=1 indicates F1 score.

    epsilon : float
        Denominator clamp minimum to prevent division by zero.
        Will be used as ``denominator.clamp(min=epsilon)``.

    Returns
    -------
    loss : torch.Tensor
        scalar loss (the smaller, the better) calculated as (1 - metric)
    """
    assert y_true.ndim == y_pred.ndim   # N, C, ... (spatial dimensions D, H, W, etc.)
    ndim = y_true.ndim
    if average == "micro":      # compute metric as if entire tensors are flattened
        axes = tuple(range(ndim))
    elif average == "samples":  # compute metric per sample, then average across samples
        axes = tuple(range(1, ndim))
    elif average == "macro":    # compute metric per channel, then average across channels
        axes = (0, ) + tuple(range(2, ndim))
    else:
        raise ValueError(f"Unsupported average method {average}")

    if metric == "precision":
        score = precision(y_true=y_true, y_pred=y_pred, reduce_axes=axes, epsilon=epsilon)
    elif metric == "recall":
        score = recall(y_true=y_true, y_pred=y_pred, reduce_axes=axes, epsilon=epsilon)
    elif metric == "f_beta":
        score = f_beta(y_true=y_true, y_pred=y_pred, reduce_axes=axes, beta=beta, epsilon=epsilon)
    elif metric == "dice":
        score = dice(y_true=y_true, y_pred=y_pred, reduce_axes=axes, beta=beta, epsilon=epsilon)
    else:
        raise ValueError(f"Unsupported metric {metric}")

    # loss is smaller the better; these metrics are bigger the better
    return 1 - score.mean()

