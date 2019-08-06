#!/usr/bin/env python

from Compiler.types import sint, cint
from library import *
from util import *


# Various, probably redundant, utility methods

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


def prod(left, right):
    """Pairwise product of elements."""
    return [l * r for l, r in zip(left, right)]


def neg(bits):
    """Bitwise not of each element in bits."""
    return [1 - bit for bit in bits]


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


# Core functionality

class Samples:

    def __init__(self, samples, n, m):
        """Create Samples object.

        :param samples: raw matrix containing continuous and discrete (binary) attribute values, category (binary),
        and indicator vector, each row represents a sample
        :param n: number of continuous attributes
        :param m: number of discrete attributes
        """
        if len(samples[0]) != n + m + 2:
            raise Exception("Wrong number of cols. in samples matrix")
        self.samples = samples
        self.n = n
        self.m = m

    def get_col(self, col_idx):
        return [row[col_idx] for row in self.samples]

    def is_cont_attribute(self, col_idx):
        if col_idx > self.n + self.m:
            raise ValueError
        return col_idx < self.n

    def __len__(self):
        return len(self.samples)


def argmax_over_fracs(elements, key_col_idx=0, val_col_idx=1):
    """Computes argmax of elements.

    Assumes elements has two columns, a key column and value column. Key column is required to be fractions,
    represented as tuples.

    :param elements: represented as tuples of numerator and denominator.
    :param key_col_idx: column to do max over
    :param val_col_idx: column to return value from
    :return tuple of index and max fraction
    """

    def _select_larger(tup_a, tup_b):
        num_a, denom_a = tup_a[key_col_idx]
        num_b, denom_b = tup_b[key_col_idx]

        lt = (num_a * denom_b) > (num_b * denom_a)
        updated_val = if_else(lt, tup_a[val_col_idx], tup_b[val_col_idx])
        updated_num = if_else(lt, num_a, num_b)
        updated_denom = if_else(lt, denom_a, denom_b)
        return (updated_num, updated_denom), updated_val

    if not elements:
        raise Exception("No elements to argmax on")
    if len(elements[0]) != 2:
        raise Exception("Must have exactly two columns")

    return tree_reduce(_select_larger, elements)


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
    res = samples

    for i in range(len(samples)):
        for j in reversed(range(i)):
            res[j], res[j + 1] = cond_swap_rows(res[j], res[j + 1], key_col_idx)

    return res


def compute_cont_ginis(samples, attr_col_idx, class_col_idx, active_col_idx):
    """Computes gini values as fractions for given attribute.

    :param samples:
    :param attr_col_idx:
    :param class_col_idx:
    :param active_col_idx:
    :return:
    """
    if not samples.is_cont_attribute(attr_col_idx):
        raise Exception("Can only call this on continuous attribute")

    # TODO only use necessary columns
    # TODO Samples class is awkward
    byattr = Samples(naive_sort_by(samples.samples, attr_col_idx), samples.n, samples.m)

    class_col = byattr.get_col(class_col_idx)
    is_active = byattr.get_col(active_col_idx)
    # all samples of class 1 that are still active
    is_one = prod(class_col, is_active)
    # active 0 samples
    is_zero = prod(neg(is_one), is_active)

    fractions = []
    # we can skip last entry; the fraction here is always (0, 0) since splitting on the last attribute always
    # partitions the samples a set containing all input samples and the empty set
    for row_idx in range(len(byattr) - 1):
        denominator, numerator = _compute_gini_fraction(is_active, is_one, is_zero, row_idx)
        fractions.append((numerator, denominator))

    # include fraction for splitting on last term
    fractions.append((sint(0), sint(0)))

    return fractions


def _compute_gini_fraction(is_active, is_one, is_zero, row_idx):
    # TODO keep updating values as we go instead of recomputing sum
    leq_this = sum(is_active[:row_idx + 1])
    gt_this = sum(is_active[row_idx + 1:])

    # total rows from 0 to row_idx + 1 of class 1
    ones_leq = sum(is_one[:row_idx + 1])
    # total rows from row_idx + 1 to total_rows of class 1
    ones_gt = sum(is_one[row_idx + 1:])

    # total rows from 0 to row_idx + 1 of class 1
    zeroes_leq = sum(is_zero[:row_idx + 1])
    # total rows from row_idx + 1 to total_rows of class 1
    zeroes_gt = sum(is_zero[row_idx + 1:])

    # Note that ones_leq = |D'_{C_{attr_col_idx} <= c_{attr_col_idx, row_idx}} ^ D'_{Y = 1}|
    # where D' is D sorted by the attribute at attr_col_idx
    numerator_one_term = \
        (ones_leq ** 2) * gt_this + (ones_gt ** 2) * leq_this
    numerator_zero_term = \
        (zeroes_leq ** 2) * gt_this + (zeroes_gt ** 2) * leq_this
    numerator = numerator_one_term + numerator_zero_term
    denominator = leq_this * gt_this
    return denominator, numerator


def test():
    def runtime_assert_mat_equals(expected, actual):
        if isinstance(actual[0][0], sint):
            actual = reveal_mat(actual)
        for expected_row, actual_row in zip(expected, actual):
            for expected_val, actual_val in zip(expected_row, actual_row):
                runtime_assert_equals(expected_val, actual_val)

    def runtime_assert_equals(expected, actual):
        if isinstance(actual, sint):
            actual = actual.reveal()
        if not isinstance(actual, cint):
            raise Exception
        eq = expected == actual

        # TODO not a proper assert

        @if_e(eq)
        def _():
            print_str("Passed.\n")

        @else_
        def _():
            print_ln("Expected %s but was %s", expected, actual)

    def test_argmax():
        key, val = argmax_over_fracs([
            ((sint(1), sint(1)), 0),
            ((sint(9), sint(4)), 1),
            ((sint(2), sint(1)), 2),
            ((sint(1), sint(2)), 3),
        ])
        runtime_assert_equals(1, val)

        key, val = argmax_over_fracs([
            ((sint(1), sint(2)), 0),
            ((sint(2), sint(5)), 1),
            ((sint(2), sint(6)), 2),
            ((sint(1), sint(9)), 3),
            ((sint(9), sint(10)), 4),
        ])
        runtime_assert_equals(4, val)

    def test_naive_sort_by():
        sec_mat = input_matrix([
            [2, 0, 0],
            [3, 1, 1],
            [1, 0, 1],
            [0, 1, 0]
        ])
        actual = naive_sort_by(sec_mat, 0)
        runtime_assert_mat_equals(
            [[0, 1, 0],
             [1, 0, 1],
             [2, 0, 0],
             [3, 1, 1]],
            actual)

    def test_compute_cont_ginis():
        sec_mat = input_matrix([
            [3, 0, 1],
            [1, 1, 1],
            [2, 1, 1]
        ])
        actual = compute_cont_ginis(Samples(sec_mat, 1, 0), 0, 1, 2)
        runtime_assert_mat_equals([(4, 2), (6, 2), (0, 0)], actual)

    test_argmax()
    test_naive_sort_by()
    test_compute_cont_ginis()


test()
