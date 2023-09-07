import torch


def pearson_corr(y_pred, y_true):
    """
    Compute the Pearson correlation coefficient between true and predicted values.

    :param y_pred: torch.Tensor
        The predicted target values from the model. Should be a 1D tensor of shape (n_samples,).
    :param y_true: torch.Tensor
        The true target values. Should be a 1D tensor of shape (n_samples,).
    :return: torch.Tensor
        The Pearson correlation coefficient. A single scalar value between -1 and 1.

    """
    vx = y_true - torch.mean(y_true)
    vy = y_pred - torch.mean(y_pred)
    corr = torch.sum(vx * vy) / (torch.sqrt(torch.sum(vx ** 2)) * torch.sqrt(torch.sum(vy ** 2)))
    return corr
