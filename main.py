#!/usr/bin/env python
import sys

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

MPC_ERROR_FLAG = "MPC_ERROR"


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

    def get_class_col_idx(self):
        return self.n + self.m

    def get_active_col_idx(self):
        return self.n + self.m + 1

    def __len__(self):
        return len(self.samples)


def argmax_over_fracs(elements):
    """Computes argmax of elements.

    Assumes elements has three columns, a numerator column, denominator col., and value column.

    NOTE: undefined if denominator of any fraction is 0.

    :param elements: represented as tuples of numerator, denominator, value
    :return tuple of index and max fraction
    """

    def debug_sanity_check(d):
        # INSECURE for debugging only!
        # Verifies that denominator is not 0
        @if_((d == 0).reveal())
        def _():
            print_ln("%s 0 in denominator.", MPC_ERROR_FLAG)

    def _select_larger(tup_a, tup_b):
        num_a, denom_a, val_a = tup_a
        num_b, denom_b, val_b = tup_b

        debug_sanity_check(denom_a)
        debug_sanity_check(denom_b)

        a_gt = (num_a * denom_b) > (num_b * denom_a)
        updated_val = if_else(a_gt, val_a, val_b)
        updated_num = if_else(a_gt, num_a, num_b)
        updated_denom = if_else(a_gt, denom_a, denom_b)
        return updated_num, updated_denom, updated_val

    if not elements:
        raise Exception("No elements to argmax on")
    if len(elements[0]) != 3:
        raise Exception("Must have exactly three columns")

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


def compute_cont_ginis(samples, attr_col_idx):
    """Computes gini values as fractions for given attribute.

    :param samples:
    :param attr_col_idx:
    :return:
    """
    if not samples.is_cont_attribute(attr_col_idx):
        raise Exception("Can only call this on continuous attribute")

    class_col_idx = samples.get_class_col_idx()
    active_col_idx = samples.get_active_col_idx()

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
        threshold = byattr.samples[row_idx][attr_col_idx]
        denominator, numerator = _compute_gini_fraction(is_active, is_one, is_zero, row_idx)
        fractions.append((numerator, denominator, threshold))

    # include fraction for splitting on last term
    last_threshold = byattr.samples[-1][attr_col_idx]
    fractions.append((sint(0), sint(1), last_threshold))

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


def compute_best_gini_cont(samples, attr_col_idx):
    """Computes best gini for given attribute.

    :param samples:
    :param attr_col_idx:
    :return:
    """
    cand_ginis = compute_cont_ginis(samples, attr_col_idx)
    return argmax_over_fracs(cand_ginis)


def test():
    def default_test_name():
        return sys._getframe(1).f_code.co_name

    def runtime_assert_arr_equals(expected, actual, test_name):
        if isinstance(actual[0], sint):
            actual = [v.reveal() for v in actual]

        if len(expected) != len(actual):
            print_ln("%s in %s dimensions wrong", MPC_ERROR_FLAG, test_name)

        for expected_val, actual_val in zip(expected, actual):
            runtime_assert_equals(expected_val, actual_val, test_name)

    def runtime_assert_mat_equals(expected, actual, test_name):
        if isinstance(actual[0][0], sint):
            actual = reveal_mat(actual)
        if len(expected) != len(actual):
            print_ln("%s in %s dimensions wrong.", MPC_ERROR_FLAG, test_name)

        for expected_row, actual_row in zip(expected, actual):
            runtime_assert_arr_equals(expected_row, actual_row, test_name)

    def runtime_assert_equals(expected, actual, test_name):
        if isinstance(actual, sint):
            actual = actual.reveal()
        if not isinstance(actual, cint):
            raise Exception
        eq = expected == actual

        # TODO not a proper assert

        @if_e(eq)
        def _():
            print_ln("%s passed.", test_name)

        @else_
        def _():
            print_ln("%s in %s.Expected %s but was %s", MPC_ERROR_FLAG, test_name, expected, actual)

    def test_argmax():
        sec_mat = input_matrix([
            [1, 1, 0],
            [9, 4, 1],
            [2, 1, 2],
            [1, 2, 3]
        ])
        num, denom, val = argmax_over_fracs(sec_mat)
        runtime_assert_equals(1, val, default_test_name())

        sec_mat = input_matrix([
            [1, 2, 0],
            [2, 5, 1],
            [2, 6, 2],
            [1, 9, 3],
            [9, 10, 4]
        ])
        num, denom, val = argmax_over_fracs(sec_mat)
        runtime_assert_equals(4, val, default_test_name())

        sec_mat = input_matrix([
            [1, 9, 0],
            [0, 1, 1]
        ])
        num, denom, val = argmax_over_fracs(sec_mat)
        runtime_assert_equals(0, val, default_test_name())

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
            actual, default_test_name())

    def test_compute_cont_ginis():
        sec_mat = input_matrix([
            [3, 0, 1],
            [1, 1, 1],
            [2, 1, 1]
        ])
        actual = compute_cont_ginis(Samples(sec_mat, 1, 0), 0)
        runtime_assert_mat_equals([(4, 2, 1), (6, 2, 2), (0, 1, 3)], actual, default_test_name())

    def test_compute_best_gini_cont():
        sec_mat = input_matrix([
            [3, 0, 1],
            [1, 1, 1],
            [2, 1, 1]
        ])
        actual = compute_best_gini_cont(Samples(sec_mat, 1, 0), 0)
        runtime_assert_arr_equals([6, 2, 2], actual, default_test_name())

    test_argmax()
    test_naive_sort_by()
    test_compute_cont_ginis()
    test_compute_best_gini_cont()


test()
