import torch
import torch.nn as nn
import torch.optim as optim
import torch.func as func

def svm_loss(xx, yy, weight, bias, c):
    """Perform the loss of SGD SVM
    Inspired by https://github.com/kazuto1011/svm-pytorch

    Args:
        xx (Tensor): (batch_size, input_dim)
        yy (Tensor): (batch_size, 1)
        weight (Tensor): (input_dim)
        bias (Tensor): (1)

    Returns:
        loss (float)
    """

    output = xx @ weight.t() + bias

    hinge_loss = torch.clamp(1 - yy * output, min=0)
    masked_tensor = torch.where(yy == 0, torch.nan, hinge_loss)
    mean = torch.nanmean(masked_tensor, dim=-1)

    loss = torch.where(mean.isnan(), torch.tensor(0.0, device=mean.device), mean)
    loss += c * (weight.t() @ weight) / 2.0

    return loss


def batch_fit_linear_svm(X, Y, lr=0.1, epoch=10, device='cpu', batchsize=5, c=0.01):
    """Perform a batch of weighted linear regression

    Args:
        X (Tensor): (n_parameter_set, batch_size, input_dim)
        y (Tensor): (n_parameter_set, batch_size, output_dim)

    Returns:
        Tensor: (n_parameter_set, input_dim + 1)
    """

    b, m, n = X.shape

    weights = torch.randn(b, n, device=device, requires_grad=True)
    biases = torch.randn(b, 1, device=device, requires_grad=True)

    X, Y = X.to(device), Y.to(device).squeeze(-1)

    for e in range(epoch):
        perm = torch.randperm(m, device=device)

        for i in range(0, m, batchsize):
            xx = X[:, perm[i : i + batchsize], :] 
            yy = Y[:, perm[i : i + batchsize]]  
            
            grads_w, grads_b = func.vmap(func.grad(svm_loss,argnums=(2,3)), in_dims=(0, 0, 0, 0, None))(*(xx, yy, weights, biases, c))

            with torch.no_grad():
                weights -= lr * grads_w
                biases -= lr * grads_b

    return torch.cat((weights.detach(), biases.detach()),dim=1)


def predict_linear_svm(X, parameters):
    """Perform a linear transformation

    Args:
        X (Tensor): (batch_size, input_dim)
        parameters (Tensor): (input_dim + 1, 1)

    Returns:
        Tensor: (batch_size, 1)
    """

    params = parameters.squeeze()
    w = params[:-1]
    b = params[-1]

    y = X @ w + b

    signs = torch.sign(y).view(-1, 1)
    return torch.where(signs == 0, 1, signs)


_batch_predict_linear_svm = torch.vmap(
    predict_linear_svm, in_dims=(None, 0)
)


def batch_predict_linear_svm(X, parameters):
    """Perform a linear transformation with a batch of parameters

    Args:
        X (Tensor): (n_parameter_set, batch_size, input_dim)
        parameters (Tensor): (n_parameter_set, input_dim + 1, 1)

    Returns:
        Tensor: (n_parameter_set, batch_size, 1)
    """
    return _batch_predict_linear_svm(X, parameters)
