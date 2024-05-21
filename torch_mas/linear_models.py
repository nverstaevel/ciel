import torch


def fit_linear_regression(X, y, weights, l1_penalty=0.1):
    """Perform a weighted linear regression

    Args:
        X (Tensor): (batch_size, input_dim)
        y (Tensor): (batch_size, output_dim)
        weights (Tensor): (batch_size, 1)

    Returns:
        Tensor: (input_dim + 1, output_dim)
    """
    num_samples = X.size(0)
    X_with_bias = torch.cat((X, torch.ones((num_samples, 1))), dim=-1)
    W = torch.diag(weights)
    X_weighted = W @ X_with_bias
    y_weighted = W @ y
    # Concatenate a column of ones to X for the bias term
    # Compute parameters
    identity_matrix = torch.eye(X_weighted.size(1))
    parameters = torch.linalg.lstsq(
        X_weighted.T @ X_weighted + l1_penalty * identity_matrix,
        X_weighted.T @ y_weighted,
    ).solution
    return parameters


_batch_fit_linear_regression = torch.vmap(
    fit_linear_regression, in_dims=(0, 0, 0, None)
)


def batch_fit_linear_regression(X, y, weights, l1_penalty=0.1):
    """Perform a batch of weighted linear regression

    Args:
        X (Tensor): (n_parameter_set, batch_size, input_dim)
        y (Tensor): (n_parameter_set, batch_size, output_dim)
        weights (Tensor): (n_parameter_dim, batch_size, 1)

    Returns:
        Tensor: (n_parameter_set, input_dim + 1, output_dim)
    """
    return _batch_fit_linear_regression(X, y, weights, l1_penalty)


def predict_linear_regression(X, parameters):
    """Perform a linear transformation

    Args:
        X (Tensor): (batch_size, output_dim)
        parameters (Tensor): (input_dim + 1, output_dim)

    Returns:
        Tensor: (batch_size, output_dim)
    """
    return torch.cat((X, torch.ones((X.size(0), 1))), dim=-1) @ parameters


_batch_predict_linear_regression = torch.vmap(
    predict_linear_regression, in_dims=(None, 0)
)


def batch_predict_linear_regression(X, parameters):
    """Perform a linear transformation with a batch of parameters

    Args:
        X (Tensor): (n_parameter_set, batch_size, output_dim)
        parameters (Tensor): (n_parameter_set, input_dim + 1, output_dim)

    Returns:
        Tensor: (n_parameter_set, batch_size, output_dim)
    """
    return _batch_predict_linear_regression(X, parameters)
