import torch
import itertools
import numpy as np


def addition_mod_p_data(p, eq_token, op_token):
    """x + y (mod p) for 0 ≤ x, y < p. Returns (5, p²) tensor."""
    x = torch.arange(p)
    y = torch.arange(p)
    x, y = torch.cartesian_prod(x, y).T
    result = (x + y) % p
    return torch.stack([x, torch.full_like(x, op_token), y, torch.full_like(x, eq_token), result])


def subtraction_mod_p_data(p, eq_token, op_token):
    """x - y (mod p) for 0 ≤ x, y < p. Returns (5, p²) tensor."""
    x = torch.arange(p)
    y = torch.arange(p)
    x, y = torch.cartesian_prod(x, y).T
    result = (x - y) % p
    return torch.stack([x, torch.full_like(x, op_token), y, torch.full_like(x, eq_token), result])


def multiplication_mod_p_data(p, eq_token, op_token):
    """x * y (mod p) for 0 ≤ x < p, 0 < y < p. Returns (5, p(p-1)) tensor."""
    x = torch.arange(p)
    y = torch.arange(1, p)
    x, y = torch.cartesian_prod(x, y).T
    result = x * y % p
    return torch.stack([x, torch.full_like(x, op_token), y, torch.full_like(x, eq_token), result])


def division_mod_p_data(p, eq_token, op_token):
    """x / y (mod p) via Fermat inverse for 0 ≤ x < p, 0 < y < p. Returns (5, p(p-1)) tensor."""
    x = torch.arange(p)
    y = torch.arange(1, p)
    x, y = torch.cartesian_prod(x, y).T
    y_inv = torch.tensor([pow(int(yi), p - 2, p) for yi in y])
    result = (x * y_inv) % p
    return torch.stack([x, torch.full_like(x, op_token), y, torch.full_like(x, eq_token), result])


def permutation_s5_data(eq_token, op_token):
    """S_5 composition: a ○ b = b[a]. Returns (5, 14400) tensor."""
    perms = [np.array(p) for p in itertools.permutations(range(5))]
    perm_to_idx = {tuple(p): i for i, p in enumerate(perms)}

    x_list, y_list, result_list = [], [], []
    for a in perms:
        for b in perms:
            c = b[a]
            x_list.append(perm_to_idx[tuple(a)])
            y_list.append(perm_to_idx[tuple(b)])
            result_list.append(perm_to_idx[tuple(c)])

    x = torch.tensor(x_list)
    y = torch.tensor(y_list)
    result = torch.tensor(result_list)
    return torch.stack([x, torch.full_like(x, op_token), y, torch.full_like(x, eq_token), result])


def sparse_parity_data(n, k, eq_token, seed=42):
    """XOR of k random bits from n-bit input. Returns (n+2, 2^n) tensor."""
    rng = torch.Generator().manual_seed(seed)
    secret = torch.randperm(n, generator=rng)[:k].tolist()
    indices = torch.arange(2 ** n)
    bits = ((indices.unsqueeze(1) >> torch.arange(n - 1, -1, -1)) & 1)
    parity = bits[:, secret].sum(dim=1) % 2
    return torch.stack([*bits.T, torch.full((2 ** n,), eq_token), parity])


def multiplication_mod_p_data_shuffled(p, eq_token, op_token, seed=42):
    """Same as mul, but token indices are randomly permuted."""
    torch.manual_seed(seed)
    shuffle = torch.randperm(p)  # random mapping

    x = torch.arange(p)
    y = torch.arange(1, p)
    x, y = torch.cartesian_prod(x, y).T
    result = x * y % p

    # Apply shuffle to all value tokens
    x = shuffle[x]
    y = shuffle[y]
    result = shuffle[result]

    return torch.stack([x, torch.full_like(x, op_token), y, torch.full_like(x, eq_token), result])

