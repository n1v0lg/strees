#!/usr/bin/env python

import util as mpc_util
import library as mpc_lib
import os
import sys


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


def input_matrix(raw_mat):
    """ TODO probably already exists """
    if len(raw_mat) == 0:
        raise Exception("Empty matrix")
    if not raw_mat[0]:
        raise Exception("Empty matrix")
    num_rows = len(raw_mat)
    num_cols = len(raw_mat[0])
    mat = Matrix(num_rows, num_cols, sint)
    for r in range(num_rows):
        for c in range(num_cols):
            mat[r][c] = sint(raw_mat[r][c])
    return mat


class Samples():

    def __init__(self, samples, m, n):
        """Create Samples object.

        samples -- raw matrix containing continuous and discrete (binary) attribute values, 
        and category (binary) 
        m -- number of continuous attributes
        n -- number of discrete attributes
        """
        if len(samples) != n + m + 1:
            raise Exception("Wrong number of cols. in samples matrix")
        self.samples = samples
        self.n = n
        self.m = m

    def get_col(self, col_idx):
        return [row[col_idx] for row in self.samples]


def argmax_fracs(fracs):
    def _select_larger(tup_a, tup_b):
        num_a, denom_a = tup_a[1]
        num_b, denom_b = tup_b[1]

        lt = (num_a * denom_b) > (num_b * denom_a)
        updated_idx = mpc_util.if_else(lt, tup_a[0], tup_b[0])
        updated_num = mpc_util.if_else(lt, num_a, num_b)
        updated_denom = mpc_util.if_else(lt, denom_a, denom_b)
        return (updated_idx, (updated_num, updated_denom))

    with_indexes = list([(cint(idx), el) for idx, el in enumerate(fracs)])
    return mpc_util.tree_reduce(_select_larger, with_indexes)[0]


def get_comparison_mat(elements):
    mat = Matrix(len(elements), len(elements), sint)
    for outer_idx, outer_el in enumerate(elements):
        for inner_idx, inner_el in enumerate(elements):
            if outer_idx == inner_idx:
                # a <= a is true
                mat[outer_idx][inner_idx] = sint(1)
            else:
                mat[outer_idx][inner_idx] = (outer_el <= inner_el)
    return mat


def get_comparison_mats(samples):
    comp_mats = dict()
    for cont_attr in range(samples.m):
        comp_mats[cont_attr] = get_comparison_mat(samples.get_col(cont_attr))
    return comp_mats


def prod(left, right):
    return [l * r for l, r in zip(left, right)]


def inner_prod(left, right):
    return sum(prod(left, right))


def bit_not(bits):
    return [1 - b for b in bits]


def best_gini_idx(samples, comp_mats, active_samples):

    def thresh_helper(category_vec, active_samples, comp_row):
        above_threshold_cat = inner_prod(
            comp_row, prod(active_samples, category_vec))
        above_threshold = inner_prod(comp_row, active_samples)
        return (above_threshold_cat, above_threshold)

    def compute_cont_attr_fracs(samples, comp_mats, active_samples):
        N = len(samples.samples)
        category_vec_1 = samples.get_col(N - 1)
        category_vec_0 = bit_not(category_vec_1)

        fracs = []
        # TODO isn't this matrix multiplication?
        for k in range(N):
            for i in range(samples.m):
                comp_row = comp_mats[i][k]
                comp_row_not = bit_not(comp_row)

                above_threshold_cat_1, above_threshold = \
                    thresh_helper(category_vec_1, active_samples, comp_row)

                below_threshold_cat_1, below_threshold = \
                    thresh_helper(category_vec_1, active_samples, comp_row_not)

                above_threshold_cat_0 = inner_prod(
                    comp_row, prod(active_samples, category_vec_0))

                below_threshold_cat_0 = inner_prod(
                    comp_row_not, prod(active_samples, category_vec_0))

                # TODO check if squaring is properly supported
                numerator = (above_threshold_cat_1**2) * below_threshold + \
                    (below_threshold_cat_1**2) * above_threshold
                numerator += (above_threshold_cat_0**2) * below_threshold + \
                    (below_threshold_cat_0**2) * above_threshold

                denom = above_threshold * below_threshold
                fracs.append((numerator, denom))
        return fracs

    fracs = compute_cont_attr_fracs(samples, comp_mats, active_samples)
    best_gini = argmax_fracs(fracs)
    return best_gini


def stree(samples):
    active_samples = [1 for _ in samples.samples]
    comp_mats = get_comparison_mats(samples)

    current_gini = best_gini_idx(samples, comp_mats, active_samples)


def test():
    def runtime_assert_equals(expected, actual):
        if isinstance(actual, sint):
            actual = actual.reveal()
        if not isinstance(actual, cint):
            raise NotImplemented
        eq = expected == actual
        # TODO not a proper assert
        @if_e(eq.reveal())
        def _():
            print_str("Passed.\n")
        @else_
        def _():
            print_ln("Expected %s but was %s", expected, actual)


    # TODO runtime asserts
    actual = argmax_fracs([
        (sint(1), sint(1)),
        (sint(9), sint(4)),
        (sint(2), sint(1)),
        (sint(1), sint(2)),
    ])
    runtime_assert_equals(1, actual)

    actual = argmax_fracs([
        (sint(1), sint(2)),
        (sint(2), sint(5)),
        (sint(2), sint(6)),
        (sint(1), sint(9)),
        (sint(9), sint(10)),
    ])
    runtime_assert_equals(4, actual)


def main():
    def get_samples():
        samples_mat = [
            [0, 10, 0],
            [1, 12, 0],
            [1, 30, 1]
        ]
        sec_mat = input_matrix(samples_mat)
        return Samples(samples_mat, 2, 0)

    stree(get_samples())


test()
main()
