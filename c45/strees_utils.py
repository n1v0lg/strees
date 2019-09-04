from Compiler.types import sint
from library import print_ln, print_str
from util import if_else

MPC_ERROR_FLAG = "MPC_ERROR"
MPC_WARN_FLAG = "MPC_WARN"
DEBUG = True


def debug_only(f):
    def wrapper(*args, **kwargs):
        if DEBUG:
            f(*args, **kwargs)
        return

    return wrapper


def print_list(lst):
    print_str("[ ")
    for v in lst:
        print_str("%s ", v.reveal())
    print_ln("]")


def print_mat(mat):
    print_ln("[ ")
    for row in mat:
        print_str(" ")
        print_list(row)
    print_ln("]")


def log_or(bit_a, bit_b):
    """Logical OR via arithmetic ops."""
    return bit_a + bit_b - bit_a * bit_b


def toggle(bit, elements):
    """Keeps elements same if bit is 1, sets all to 0 otherwise"""
    return [el * bit for el in elements]


def prod(left, right):
    """Pairwise product of elements."""
    return [l * r for l, r in zip(left, right)]


def inner_prod(left, right):
    """Inner product of elements."""
    return sum(l * r for l, r in zip(left, right))


def neg(bits):
    """Bitwise not of each element in bits."""
    return [1 - bit for bit in bits]


def same_len(row_a, row_b):
    if len(row_a) != len(row_b):
        raise Exception("Must be same length")


def if_else_row(bit, row_a, row_b):
    same_len(row_a, row_b)
    return [if_else(bit, a, b) for a, b in zip(row_a, row_b)]


def cond_swap_rows(row_x, row_y, key_col_idx):
    """Copied from MP-SPDZ main repo.

    Modified to support rows as entries instead of single values.
    """
    b = row_x[key_col_idx] < row_y[key_col_idx]
    row_bx = [b * x for x in row_x]
    row_by = [b * y for y in row_y]
    new_row_x = [bx + y - by for y, bx, by in zip(row_y, row_bx, row_by)]
    new_row_y = [x - bx + by for x, bx, by in zip(row_x, row_bx, row_by)]
    return new_row_x, new_row_y


def naive_sort_by(samples, key_col_idx):
    """Sorts 2D-array by specified column.

    Note: this uses naive bubble-sort.
    Copied from MP-SPDZ main repo.
    """
    res = samples[:]

    for i in range(len(samples)):
        for j in reversed(range(i)):
            res[j], res[j + 1] = cond_swap_rows(res[j], res[j + 1], key_col_idx)

    return res


def mat_assign_op(raw_mat, f):
    if len(raw_mat) == 0:
        raise Exception("Empty matrix")
    if not raw_mat[0]:
        raise Exception("Empty matrix")
    num_rows = len(raw_mat)
    num_cols = len(raw_mat[0])
    mat = []
    for r in range(num_rows):
        row = []
        for c in range(num_cols):
            row.append(f(raw_mat[r][c]))
        mat.append(row)
    return mat


def reveal_mat(mat):
    """Reveals matrix of sints.

    TODO probably already exists somewhere
    TODO use Matrix?
    """
    return mat_assign_op(mat, lambda x: x.reveal())


def input_matrix(mat):
    """Inputs matrix of ints into MPC.

    TODO probably already exists somewhere
    TODO use Matrix?
    """
    return mat_assign_op(mat, lambda x: sint(x))


def enumerate_rows(rows):
    """Adds index to end of each row."""
    return [row + [i] for i, row in enumerate(rows)]


def enumerate_vals(rows):
    """Adds index to end of each val."""
    return [[val, i] for i, val in enumerate(rows)]


def get_col(rows, col_idx):
    """Returns column at index as list."""
    return [row[col_idx] for row in rows]


def reveal_list(lst):
    """Reveals list of sints.

    TODO probably already exists somewhere.
    """
    return [val.reveal() for val in lst]


def input_list(lst):
    """Inputs list of values into MPC."""
    return [sint(val) for val in lst]


def is_two_pow(n):
    """True if 2 power.

    Lazily stolen from SO: https://stackoverflow.com/questions/57025836/check-if-a-given-number-is-power-of-two-in
    -python."""
    return (n & (n - 1) == 0) and n != 0
