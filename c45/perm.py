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


def sort_and_permute(rows, attr_idx):
    """Sorts and permutes rows according to specified permutation."""
    if not is_two_pow(len(rows)):
        raise Exception("Only powers of two supported for shuffles")

    sorted_value_col, order_col = sort_by(get_col(rows, 0), get_col(rows, 1))
     # = get_col(sorted_rows, 0)
     # = get_col(sorted_rows, 1)

    config_bits = rec_config_shuffle(rows)
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
