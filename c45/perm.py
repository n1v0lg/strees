from Compiler.types import sint, Array
from permutation import configure_waksman, random_perm, rec_shuffle

# from library import for_range_parallel, function_block, for_range

# Make IDE happy
try:
    from strees_utils import *
except ImportError:
    pass


def rec_config_shuffle(rows):
    """Configures waksman network for recursive version of the algorithm."""
    return configure_waksman(random_perm(len(rows)))


def shuffle_rows(rows, config, reverse=False):
    """Shuffles rows, according to configured permutation.

    Applies same shuffle to each column, recombines into result.
    :param rows:
    :param config:
    :param reverse:
    :return:
    """
    shuffled_cols = []
    for col_idx in range(len(rows[0])):
        col = get_col(rows, col_idx)
        rec_shuffle(col, config=config, value_type=sint, reverse=reverse)
        shuffled_cols.append(col)
    shuffled_rows = []
    for row_idx in range(len(rows)):
        shuffled_rows.append([col[row_idx] for col in shuffled_cols])
    return shuffled_rows


def sort_and_permute(rows, attr_idx):
    """Sorts and permutes rows according to specified permutation."""
    if not is_two_pow(len(rows)):
        raise Exception("Only powers of two supported for shuffles")
    config_bits = rec_config_shuffle(rows)
    sorted_rows = sort_by(rows, attr_idx)
    shuffled_rows = shuffle_rows(sorted_rows, config=config_bits)

    return shuffled_rows, config_bits


def open_permute(values, open_perm):
    """Applies a public permutation to an Array of sints."""
    # TODO find way to avoid this
    if not isinstance(values, Array):
        values = Array(len(values), sint).create_from(values)
    reordered = Array(len(values), sint)
    for idx, val in enumerate(values):
        old_idx = open_perm[idx]
        reordered[idx] = values[old_idx]
    return reordered
