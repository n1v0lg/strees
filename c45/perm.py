from Compiler.types import sint, Array
from permutation import configure_waksman, random_perm, rec_shuffle

# Make IDE happy
try:
    from strees_utils import *
except ImportError:
    pass


def rec_config_shuffle(rows):
    """Configures waksman network for recursive version of the algorithm."""
    return configure_waksman(random_perm(len(rows)))


def sort_and_permute(key_col, val_col):
    """Sorts and permutes columns."""
    same_len(key_col, val_col)
    if not is_two_pow(len(key_col)):
        raise Exception("Only powers of two supported for shuffles")

    sorted_value_col, order_col = sort_by(key_col, val_col)

    config_bits = rec_config_shuffle(key_col)
    rec_shuffle(order_col, config=config_bits, value_type=sint)

    return sorted_value_col, order_col, config_bits


def open_permute(values, open_perm):
    """Applies a public permutation to an Array of sints."""
    if not isinstance(values, Array):
        raise Exception("Must be array")
    reordered = Array(len(values), sint)
    for idx, val in enumerate(values):
        old_idx = open_perm[idx]
        reordered[idx] = values[old_idx]
    return reordered
