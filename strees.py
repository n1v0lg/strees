#!/usr/bin/env python

import util as mpc_util
import library as mpc_lib


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


def argmin(elements):
    def _select_smaller(tup_a, tup_b):
        lt = tup_a[1] < tup_b[1]
        updated_idx = mpc_util.if_else(lt, tup_a[0], tup_b[0])
        updated_val = mpc_util.if_else(lt, tup_a[1], tup_b[1])
        return (updated_idx, updated_val)

    with_indexes = list([(cint(idx), el) for idx, el in enumerate(elements)])
    return mpc_util.tree_reduce(_select_smaller, with_indexes)[0]


def get_comparison_mat(elements):
    mat = Matrix(len(elements), len(elements), sint)
    for outer_idx, outer_el in enumerate(elements):
        for inner_idx, inner_el in enumerate(elements):
            if outer_idx == inner_idx:
                # a < a is false
                mat[outer_idx][inner_idx] = sint(0)
            else:
                mat[outer_idx][inner_idx] = (outer_el < inner_el)
    return mat


def main():
    pass

main()
