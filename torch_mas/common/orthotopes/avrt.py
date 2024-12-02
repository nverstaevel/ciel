import torch
from .base import *


def init_deltas_avrt(hypercube, side_frac):
    """Create initial deltas for AVRT for min and max values on each dimensions.

    Args:
        hypercube (Tensor): (n_dim, 2)
        side_frac (Tensor): (1,)

    Returns:
        Tensor: (n_dim, 2)
    """
    h_sides = sides(hypercube)
    deltas = torch.zeros_like(hypercube, device=hypercube.device)
    deltas[:, 0] = h_sides * side_frac
    deltas[:, 1] = h_sides * side_frac
    return deltas


_batch_init_deltas_avrt = torch.vmap(init_deltas_avrt, in_dims=(0, None))


def batch_init_deltas_avrt(hypercube, side_frac):
    """Create a batch of initial deltas for AVRT for min and max values on each dimensions.

    Args:
        hypercube (Tensor): (batch_size, n_dim, 2)
        side_frac (Tensor): (1,)

    Returns:
        Tensor: (batch_size, n_dim, 2)
    """
    return _batch_init_deltas_avrt(hypercube, side_frac)


def update_hypercube_avrt(hypercube, x, deltas, fb_t, fb_tm1, acoef, dcoef):
    """Updates an hypercube towards x, modifying its sides as AVRTs

    Args:
        hypercube (Tensor): (n_dim, 2)
        x (Tensor): (n_dim,)
        deltas (Tensor): (n_dim, 2)
        fb_t (Tensor): (1,) values in {-1, 0, +1}
        fb_tm1 (Tensor): (1,) values in {-1, 0, +1}
        acoef (Tensor): (1,)
        dcoef (Tensor): (1,)

    Returns:
        (Tensor, Tensor): (n_dim, 2), (n_dim, 2)
    """
    updated_hypercube = hypercube.clone().detach()
    low, high = updated_hypercube[:, 0], updated_hypercube[:, 1]
    dims_mask = (x < high) & (x > low)
    dims_mask = torch.where(
        dims_mask.all(), dims_mask, ~dims_mask
    )  # True if dim update

    coef = torch.ones_like(deltas)

    d_mask = (fb_t == 0) | (
        (fb_t != fb_tm1) & ((torch.abs(fb_t) + torch.abs(fb_tm1)) == 2)
    )
    a_mask = (fb_t == fb_tm1) & ((torch.abs(fb_t) + torch.abs(fb_tm1)) == 2)
    coef = torch.where(d_mask, dcoef, coef)
    coef = torch.where(a_mask, acoef, coef)
    updated_deltas = torch.where(
        torch.stack([dims_mask, dims_mask], dim=1), deltas * coef, deltas
    )  # (n_dim, 2)

    new_low = low - updated_deltas[:, 0] * fb_t
    new_high = high + updated_deltas[:, 1] * fb_t

    dist_low = torch.abs(low - x)
    dist_high = torch.abs(high - x)
    mask = dist_high < dist_low  # True if high update else low update

    high = torch.where(mask & dims_mask, new_high, high)
    low = torch.where(~mask & dims_mask, new_low, low)

    return torch.stack([low, high], dim=-1), updated_deltas


_batch_update_hypercube_avrt = torch.vmap(
    update_hypercube_avrt, in_dims=(0, None, 0, 0, 0, None, None)
)


def batch_update_hypercube_avrt(hypercube, x, deltas, fb_t, fb_tm1, acoef, dcoef):
    """Updates a batch of hypercubes towards x, modifying their sides as AVRTs

    Args:
        hypercube (Tensor): (batch_size, n_dim, 2)
        x (Tensor): (n_dim,)
        deltas (Tensor): (batch_size, n_dim, 2)
        fb_t (Tensor): (batch_size,) values in {-1, 0, +1}
        fb_tm1 (Tensor): (batch_size,) values in {-1, 0, +1}
        acoef (Tensor): (1,)
        dcoef (Tensor): (1,)

    Returns:
        (Tensor, Tensor): (batch_size, n_dim, 2), (batch_size, n_dim, 2)
    """
    return _batch_update_hypercube_avrt(
        hypercube, x, deltas, fb_t, fb_tm1, acoef, dcoef
    )


def expanded_hypercube_avrt(hypercube, deltas):
    """Generate the expanded version of an hypercube from AVRT deltas

    Args:
        hypercube (Tensor): (n_dim, 2)
        deltas (Tensor): (n_dim, 2)

    Returns:
        Tensor: (n_dim, 2)
    """
    updated_hypercube = hypercube.clone().detach()
    updated_hypercube[:, 0] -= deltas[:, 0]
    updated_hypercube[:, 1] += deltas[:, 1]
    return updated_hypercube


_batch_expanded_hypercube_avrt = torch.vmap(expanded_hypercube_avrt, in_dims=(0, 0))


def batch_expanded_hypercube_avrt(hypercubes, deltas):
    """Generate the expanded version of a batch of hypercubes from AVRT deltas

    Args:
        hypercubes (Tensor): (batch_size, n_dim, 2)
        deltas (Tensor): (batch_size, n_dim, 2)

    Returns:
        Tensor: (batch_size, n_dim, 2)
    """
    return _batch_expanded_hypercube_avrt(hypercubes, deltas)


def check_hypercube(hypercube):
    """Check if low < high on at least one dimension

    Args:
        hypercube (Tensor): (n_dim, 2)

    Returns:
        BoolTensor: (1,)
    """
    return (hypercube[:, 0] > hypercube[:, 1]).any()


_batch_check_hypercube = torch.vmap(check_hypercube)


def batch_check_hypercube(hypercubes):
    """Check conformity of batch of hypercubes

    Args:
        hypercubes (Tensor): (batch_size, n_dim, 2)

    Returns:
        BoolTensor: (batch_size,)
    """
    return _batch_check_hypercube(hypercubes)
