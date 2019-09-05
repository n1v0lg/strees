from Compiler.types import sint, Matrix
from library import print_ln, print_str
from permutation import odd_even_merge_sort

MPC_ERROR_FLAG = "MPC_ERROR"
MPC_WARN_FLAG = "MPC_WARN"
DEBUG = True

# Parameter for scaling denominator in GINI index computation
ALPHA = 10


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


def alpha_scale(val):
    """Scales value according to ALPHA param."""
    return ALPHA * val + 1


def log_or(bit_a, bit_b):
    """Logical OR via arithmetic ops."""
    return bit_a + bit_b - bit_a * bit_b


def toggle(bit, elements):
    """Keeps elements same if bit is 1, sets all to 0 otherwise"""
    return [el * bit for el in elements]


def prod(left, right):
    """Pairwise product of elements."""
    same_len(left, right)
    return [l * r for l, r in zip(left, right)]


def inner_prod(left, right):
    """Inner product of elements."""
    same_len(left, right)
    if not left:
        return []
    cls = left[0].get_type(0)
    return cls.dot_product(left, right)


def neg(bits):
    """Bitwise not of each element in bits."""
    return [1 - bit for bit in bits]


def same_len(row_a, row_b):
    if len(row_a) != len(row_b):
        raise Exception("Must be same length but was {} and {}".format(len(row_a), len(row_b)))


def if_else_row(bit, row_a, row_b):
    same_len(row_a, row_b)
    return [bit.if_else(a, b) for a, b in zip(row_a, row_b)]


def sort_by(samples, key_col_idx):
    """Sorts 2D-array by specified column."""
    res = samples[:]
    odd_even_merge_sort(res, lambda a, b: a[key_col_idx] < b[key_col_idx])
    return res


def _mat_assign_op(raw_mat, f):
    if len(raw_mat) == 0:
        raise Exception("Empty matrix")
    if not raw_mat[0]:
        raise Exception("Empty matrix")
    num_rows = len(raw_mat)
    num_cols = len(raw_mat[0])

    mat = Matrix(num_rows, num_cols, sint)
    for r in range(num_rows):
        for c in range(num_cols):
            mat[r][c] = f(raw_mat[r][c])
    return mat


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
