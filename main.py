#!/usr/bin/env python

from Compiler.types import sint, cint
from library import *
from util import *


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


class Samples:

    def __init__(self, samples, n, m):
        """Create Samples object.

        :param samples: raw matrix containing continuous and discrete (binary) attribute values, category (binary),
        and indicator vector, each row represents a sample
        :param n: number of continuous attributes
        :param m: number of discrete attributes
        """
        if len(samples) != n + m + 2:
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


def argmax_fracs(fracs):
    """Computes argmax of fractions.

    :param fracs: represented as tuples of numerator and denominator.
    :return tuple of index and max fraction
    """

    def _select_larger(tup_a, tup_b):
        num_a, denom_a = tup_a[1]
        num_b, denom_b = tup_b[1]

        lt = (num_a * denom_b) > (num_b * denom_a)
        updated_idx = if_else(lt, tup_a[0], tup_b[0])
        updated_num = if_else(lt, num_a, num_b)
        updated_denom = if_else(lt, denom_a, denom_b)
        return updated_idx, (updated_num, updated_denom)

    with_indexes = list([(cint(idx), el) for idx, el in enumerate(fracs)])
    return tree_reduce(_select_larger, with_indexes)


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
        idx, val = argmax_fracs([
            (sint(1), sint(1)),
            (sint(9), sint(4)),
            (sint(2), sint(1)),
            (sint(1), sint(2)),
        ])
        runtime_assert_equals(1, idx)

        idx, val = argmax_fracs([
            (sint(1), sint(2)),
            (sint(2), sint(5)),
            (sint(2), sint(6)),
            (sint(1), sint(9)),
            (sint(9), sint(10)),
        ])
        runtime_assert_equals(4, idx)

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

    test_argmax()
    test_naive_sort_by()


test()