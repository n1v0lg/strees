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


def get_samples():
    samples_mat = [
        [0, 10, 0],
        [1, 12, 0],
        [1, 3, 0]
    ]
    sec_mat = input_matrix(samples_mat)
    return Samples(samples_mat, 1, 1)


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


def stree(samples):
    comp_mats = dict()
    for cont_attr in range(samples.m):
        comp_mats[cont_attr] = get_comparison_mat(samples.get_col(cont_attr))
        print_mat(comp_mats[cont_attr])

def main():
    stree(get_samples())


main()
