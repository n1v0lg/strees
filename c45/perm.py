import math
import random as rand

from Compiler.types import sint, Array
from library import for_range
from permutation import configure_waksman, rec_shuffle, shuffle

rand.seed(42)

# Make IDE happy
try:
    from strees_utils import *
except ImportError:
    pass


# Taken from MP-SPDZ to fix seed for rand
def random_perm(n):
    """ Generate a random permutation of length n

    WARNING: randomness fixed at compile-time, this is NOT secure
    """
    a = range(n)
    for i in range(n - 1, 0, -1):
        j = rand.randint(0, i)
        t = a[i]
        a[i] = a[j]
        a[j] = t
    return a


def config_shuffle_given_perm(perm, value_type=sint):
    """ Compute config for oblivious shuffling.

    Take mod 2 for active sec. """
    n = len(perm)
    if n & (n - 1) != 0:
        # pad permutation to power of 2
        m = 2 ** int(math.ceil(math.log(n, 2)))
        perm += range(n, m)
    config_bits = configure_waksman(perm)
    # 2-D array
    config = Array(len(config_bits) * len(perm), value_type.reg_type)
    for i, c in enumerate(config_bits):
        for j, b in enumerate(c):
            config[i * len(perm) + j] = b
    return config


def config_shuffle_for_length(n, value_type=sint):
    """ Compute config for oblivious shuffling.

    Take mod 2 for active sec. """
    perm = random_perm(n)
    return config_shuffle_given_perm(perm, value_type)


def default_config_shuffle(values, use_iter=True):
    """Configures waksman network for default shuffle algorithm."""
    if use_iter:
        return config_shuffle_for_length(len(values))
    else:
        return configure_waksman(random_perm(len(values)))


def default_shuffle(values, config, reverse=False, use_iter=True):
    """Shuffles values in place using default shuffle algorithm."""
    if use_iter:
        shuffle(values, config=config, value_type=sint, reverse=reverse)
    else:
        rec_shuffle(values, config=config, value_type=sint, reverse=reverse)


def sort_and_permute(key_col, val_col):
    """Sorts and permutes columns."""
    same_len(key_col, val_col)
    if not is_two_pow(len(key_col)):
        raise Exception("Only powers of two supported for shuffles")

    sorted_value_col, order_col = sort_by(key_col, val_col)

    config_bits = default_config_shuffle(key_col)
    default_shuffle(order_col, config=config_bits)

    return sorted_value_col, order_col, config_bits


def open_permute(values, open_perm):
    """Applies a public permutation to an Array of sints."""
    array_check(values)
    reordered = Array(len(values), sint)

    for idx in range(len(values)):
        old_idx = open_perm[idx]
        reordered[idx] = values[old_idx]

    return reordered
