#!/usr/bin/env python

import sys
from collections import deque

from Compiler.types import sint, cint, MemValue
from library import print_ln, print_str, if_, if_e, else_, do_while
from util import tree_reduce, if_else


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
    res = samples

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


# Core functionality

MPC_ERROR_FLAG = "MPC_ERROR"
MPC_WARN_FLAG = "MPC_WARN"
DEBUG = True


def debug_only(f):
    def wrapper(*args, **kwargs):
        if DEBUG:
            f(*args, **kwargs)
        return

    return wrapper


class Node:

    def __init__(self):
        self.left = None
        self.right = None

    def l(self, left):
        self.left = left
        return self

    def r(self, right):
        self.right = right
        return self


class TreeNode(Node):

    def __init__(self, is_leaf, attr_idx, threshold, is_dummy, node_class):
        """Represents tree node.

         Holds following secret flags and values.

        :param is_leaf: flag indicating whether this is a leaf node, or an internal node
        :param attr_idx: index of attribute to split on (bogus value if leaf node)
        :param threshold: threshold value to split on (bogus value if leaf node)
        :param is_dummy: flag indicating if this is a dummy leaf node (i.e., a fake leaf node that is an ancestor of
        a real leaf node)
        :param node_class: class of the node (bogus value if no leaf node)
        """
        Node.__init__(self)
        self.is_leaf = is_leaf
        self.attr_idx = attr_idx
        self.threshold = threshold
        self.is_dummy = is_dummy
        self.node_class = node_class
        self.left = None
        self.right = None

    def reveal_self(self):
        """Opens all secret values (modifies self)."""
        self.is_leaf = self.is_leaf.reveal()
        self.attr_idx = self.attr_idx.reveal()
        self.threshold = self.threshold.reveal()
        self.is_dummy = self.is_dummy.reveal()
        self.node_class = self.node_class.reveal()

    def print_self(self):
        @if_e(self.is_leaf)
        def _():
            @if_e(self.is_dummy)
            def _():
                print_str("(D)")

            @else_
            def _():
                print_str("(%s)", self.node_class)

        @else_
        def _():
            print_str("(c_{%s} <= %s)", self.attr_idx, self.threshold)


class Tree:

    def __init__(self, root):
        self.root = root

    def _reveal(self, node):
        if node:
            node.reveal_self()
            self._reveal(node.left)
            self._reveal(node.right)

    def reveal_self(self):
        self._reveal(self.root)

    @staticmethod
    def _bfs_print(node):
        queue = deque([node])
        while queue:
            curr = queue.popleft()
            print_str(" ")
            if curr:
                curr.print_self()
                queue.append(curr.left)
                queue.append(curr.right)
            else:
                print_str("(X)")
        print_ln("")

    def print_self(self):
        self._bfs_print(self.root)

    def _num_nodes(self, node):
        if node is None:
            return 0
        else:
            return 1 + self._num_nodes(node.left) + self._num_nodes(node.right)

    def num_nodes(self):
        return self._num_nodes(self.root)


class Samples:

    def __init__(self, samples, n, m=0):
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

    def get_active_col(self):
        return self.get_col(self.get_active_col_idx())

    def get_class_col(self):
        return self.get_col(self.get_class_col_idx())

    def with_updated_actives(self, updated_actives):
        # TODO hacky
        if len(updated_actives) != len(self):
            raise Exception("Incorrect number of values")
        new_samples = []
        for active_flag, row in zip(updated_actives, self.samples):
            updated_row = row[:]
            updated_row[self.get_active_col_idx()] = active_flag
            new_samples.append(updated_row)
        return Samples(new_samples, self.n, self.m)

    def __len__(self):
        return len(self.samples)


def argmax_over_fracs(elements):
    """Computes argmax of elements.

    Supports arbitrary number of columns but assumes that numerator and denominator are the first two columns.

    NOTE: undefined if denominator of any fraction is 0.

    :param elements: represented as rows of form [numerator, denominator, ...]
    :return row with max fraction
    """

    @debug_only
    def debug_sanity_check(d):
        # INSECURE for debugging only!
        # Verifies that denominator is not 0
        @if_((d.reveal() == 0))
        def _():
            # Only a warning since this is acceptable when we are obliviously splitting even though we've already
            # reached a leaf node
            print_ln("%s 0 in denominator.", MPC_WARN_FLAG)

    def _select_larger(row_a, row_b):
        num_a, denom_a = row_a[0], row_a[1]
        num_b, denom_b = row_b[0], row_b[1]

        debug_sanity_check(denom_a)
        debug_sanity_check(denom_b)

        a_gt = (num_a * denom_b) > (num_b * denom_a)
        return if_else_row(a_gt, row_a, row_b)

    if not elements:
        raise Exception("No elements to argmax on")
    if len(elements) == 1:
        return elements[0]
    return tree_reduce(_select_larger, elements)


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
        numerator, denominator = _compute_gini_fraction(is_active, is_one, is_zero, row_idx)
        fractions.append([numerator, denominator, threshold])

    # include fraction for splitting on last term
    last_threshold = byattr.samples[-1][attr_col_idx]
    fractions.append([sint(0), sint(1), last_threshold])

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
    return numerator, denominator


def compute_best_gini_cont(samples, attr_col_idx):
    """Computes best gini for given attribute.

    :param samples:
    :param attr_col_idx:
    :return:
    """
    cand_ginis = compute_cont_ginis(samples, attr_col_idx)
    return argmax_over_fracs(cand_ginis)


def select_col_at(samples, idx):
    """Selects column at given index, obliviously if index is secret.

    :param samples:
    :param idx:
    :return:
    """

    @debug_only
    def debug_sanity_check(i):
        # INSECURE for debugging only!
        # Verifies that idx is within range
        @if_((i >= samples.n + samples.m).reveal())
        def _():
            print_ln("%s index is out of range.", MPC_ERROR_FLAG)

    if isinstance(idx, int):
        return samples.get_col(idx)

    if not isinstance(idx, sint):
        raise Exception("Only use this if index is secret or int")

    debug_sanity_check(idx)
    # TODO optimize
    res = []
    eq_flags = [idx == i for i in range(samples.n + samples.m)]
    for row in samples.samples:
        res.append(inner_prod(eq_flags, row))
    return res


def partition_on(samples, attr_idx, threshold, is_leaf):
    """Partitions samples on given attribute and threshold.

    Sets all indicator values on both partitions to 0 if we're dealing with leaf node.

    :param samples:
    :param attr_idx:
    :param threshold:
    :param is_leaf:
    :return:
    """
    selected_col = select_col_at(samples, attr_idx)

    # TODO this only works for binary discrete attributes,
    # else have to obliviously distinguish whether to use an eq or a leq
    # IDEA if we have few classes, we can represent these as via separate indicator columns, one for each class
    # this lets us avoid using equality checks
    go_left = [v <= threshold for v in selected_col]
    go_right = neg(go_left)

    # mask out rows that are already inactive
    active_col = samples.get_active_col()
    go_left = prod(go_left, active_col)
    # TODO can we derive this from go_left?
    go_right = prod(go_right, active_col)

    # set both indicator vectors to 0 if we're at a leaf node
    is_internal = 1 - is_leaf
    go_left = toggle(is_internal, go_left)
    go_right = toggle(is_internal, go_right)

    left = samples.with_updated_actives(go_left)
    right = samples.with_updated_actives(go_right)

    return left, right


def determine_if_leaf(samples):
    """Computes if this node is a leaf.

    This is the case if (i) all samples are inactive, or (ii) all active samples have the same class.

    :param samples:
    :return:
    """
    active_col = samples.get_active_col()
    is_category_one = samples.get_class_col()
    is_category_zero = neg(is_category_one)

    active_ones = prod(is_category_one, active_col)
    active_zeroes = prod(is_category_zero, active_col)

    total_actives = sum(active_col)

    all_inactive = total_actives == 0
    all_ones = sum(active_ones) == total_actives
    all_zeroes = sum(active_zeroes) == total_actives

    is_leaf = log_or(all_inactive, log_or(all_ones, all_zeroes))
    return is_leaf, all_inactive, all_ones


def c45_single_round(samples):
    """Runs single round of C4.5 algorithm.

    :param samples:
    :return:
    """
    if samples.m != 0:
        # TODO
        raise Exception("Discrete attributes not implemented yet")

    # since we don't want to leak anything apart from an upper bound on the depth of the tree,
    # we need to both compute if we have reached a leaf case, and a splitting attribute

    # Base case: compute if this node is a leaf node, if it's a dummy, and its class
    is_leaf, is_dummy, node_class = determine_if_leaf(samples)

    # compute best attribute and threshold to split on
    candidates = []
    for c in range(samples.n):
        num, denom, thresh = compute_best_gini_cont(samples, c)
        candidates.append((num, denom, c, thresh))
    _, _, attr_idx, thresh = argmax_over_fracs(candidates)

    # TODO Base case: check if partitioning on best attribute doesn't actually further partition the data
    # This can happen if it's not possible to partition the data perfectly, i.e., we end up with partitions that
    # include both class 0 and class 1 samples

    # partition samples on selected attribute
    left, right = partition_on(samples, attr_idx, thresh, is_leaf)

    # wrap index in sint, in case it isn't secret (can happen if we only have one attribute)
    attr_idx = sint(attr_idx) if isinstance(attr_idx, int) else attr_idx

    node = TreeNode(is_leaf, attr_idx, thresh, is_dummy, node_class)
    return node, left, right


def c45(input_samples, max_iteration_count):
    """Runs C4.5 algorithm to construct decision tree.

    This implementation uses an iterative approach as opposed to the more obvious recursive approach since this seems
    better aligned with MP-SPDZ.

    TODO Executes exactly max_iteration_count iterations for now

    :param input_samples:
    :param max_iteration_count: upper limit on iterations to generate decision tree for samples
    :return:
    """
    queue = deque([(None, input_samples)])
    root = None

    # Create tree in a BFS-like traversal
    for _ in range(max_iteration_count):
        parent, samples = queue.popleft()
        node, left_samples, right_samples = c45_single_round(samples)
        if parent:
            # if there is a parent, we need to link the current node back to it as a child
            # since we're doing a BFS, the left child will always come first
            if not parent.left:
                parent.left = node
            else:
                parent.right = node

        # track root node, to return it later
        if not root:
            root = node

        # push back partitioned samples to process, along with the current node which will be the resulting nodes'
        # parent
        queue.append((node, left_samples))
        queue.append((node, right_samples))

    # TODO post-process tree to fill dummy leaves that have real sibling with actual categories
    # TODO can this still ever happen? seems like the parent of those will have invariably been a leaf node already
    return Tree(root)


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

    class DN(Node):

        def __init__(self, attr_idx, threshold):
            """Decision node."""
            Node.__init__(self)
            self.attr_idx = attr_idx
            self.threshold = threshold
            self.left = None
            self.right = None

    class LN(Node):

        def __init__(self, node_class, is_dummy=False):
            """Leaf node."""
            Node.__init__(self)
            self.node_class = node_class
            self.is_dummy = is_dummy
            self.left = None
            self.right = None

    def runtime_assert_node_equals(expected, actual, test_name):
        if isinstance(expected, LN):
            runtime_assert_equals(1, actual.is_leaf, test_name)
            if expected.is_dummy:
                # dummy case
                runtime_assert_equals(1, actual.is_dummy, test_name)
                # don't test class because this is a dummy node
            else:
                runtime_assert_equals(0, actual.is_dummy, test_name)
                runtime_assert_equals(expected.node_class, actual.node_class, test_name)
        elif isinstance(expected, DN):
            runtime_assert_equals(0, actual.is_leaf, test_name)
            runtime_assert_equals(expected.attr_idx, actual.attr_idx, test_name)
            runtime_assert_equals(expected.threshold, actual.threshold, test_name)
        else:
            print_ln("%s incorrect expected tree, must only hold nodes", MPC_ERROR_FLAG)

    def runtime_assert_tree_equals(expected, actual, test_name):
        # TODO how should dummy nodes be handled?
        actual.reveal_self()
        expected_num_nodes = expected.num_nodes()
        actual_num_nodes = actual.num_nodes()
        if expected_num_nodes != actual_num_nodes:
            print_ln("%s Expected tree has %s nodes but actual has %s",
                     MPC_ERROR_FLAG,
                     expected_num_nodes,
                     actual_num_nodes)
        queue = deque([(expected.root, actual.root)])
        while queue:
            expected_next_node, actual_next_node = queue.popleft()
            if expected_next_node:
                if not actual_next_node:
                    print_ln("%s trees have different topology", MPC_ERROR_FLAG)
                runtime_assert_node_equals(expected_next_node, actual_next_node, test_name)
                queue.append((expected_next_node.left, actual_next_node.left))
                queue.append((expected_next_node.right, actual_next_node.right))
            else:
                if actual_next_node:
                    print_ln("%s trees have different topology", MPC_ERROR_FLAG)

    def test_argmax():
        sec_mat = input_matrix([
            [1, 1, 0],
            [9, 4, 1],
            [2, 1, 2],
            [1, 2, 3]
        ])
        actual = argmax_over_fracs(sec_mat)
        runtime_assert_arr_equals([9, 4, 1], actual, default_test_name())

        sec_mat = input_matrix([
            [1, 2, 0],
            [2, 5, 1],
            [2, 6, 2],
            [1, 9, 3],
            [9, 10, 4]
        ])
        actual = argmax_over_fracs(sec_mat)
        runtime_assert_arr_equals([9, 10, 4], actual, default_test_name())

        sec_mat = input_matrix([
            [1, 9, 0],
            [0, 1, 1]
        ])
        actual = argmax_over_fracs(sec_mat)
        runtime_assert_arr_equals([1, 9, 0], actual, default_test_name())

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

    def test_obl_select_col_at():
        sec_mat = input_matrix([
            [1, 2, 3, 1, 1],
            [4, 5, 6, 1, 1],
            [7, 8, 9, 1, 1]
        ])
        actual = select_col_at(Samples(sec_mat, 3, 0), sint(1))
        runtime_assert_arr_equals([2, 5, 8], actual, default_test_name())

    def test_partition_on():
        sec_mat = input_matrix([
            [1, 2, 3, 1, 1],
            [4, 5, 6, 1, 1],
            [7, 8, 9, 1, 1],
            [10, 11, 12, 1, 1]
        ])
        left, right = partition_on(Samples(sec_mat, 3, 0),
                                   attr_idx=sint(1), threshold=5, is_leaf=sint(0))
        runtime_assert_mat_equals(
            [[1, 2, 3, 1, 1],
             [4, 5, 6, 1, 1],
             [7, 8, 9, 1, 0],
             [10, 11, 12, 1, 0]],
            left.samples,
            default_test_name()
        )
        runtime_assert_mat_equals(
            [[1, 2, 3, 1, 0],
             [4, 5, 6, 1, 0],
             [7, 8, 9, 1, 1],
             [10, 11, 12, 1, 1]],
            right.samples,
            default_test_name()
        )

        sec_mat = input_matrix([
            [1, 2, 3, 1, 0],
            [4, 5, 6, 1, 1],
            [7, 8, 9, 1, 0],
            [10, 11, 12, 1, 1]
        ])
        left, right = partition_on(Samples(sec_mat, 3, 0), attr_idx=sint(1), threshold=5,
                                   is_leaf=0)
        runtime_assert_arr_equals(
            [0, 1, 0, 0],
            left.get_active_col(),
            default_test_name()
        )
        runtime_assert_arr_equals(
            [0, 0, 0, 1],
            right.get_active_col(),
            default_test_name()
        )

        sec_mat = input_matrix([
            [1, 2, 3, 1, 0],
            [4, 5, 6, 1, 1],
            [7, 8, 9, 1, 0],
            [10, 11, 12, 1, 1]
        ])
        left, right = partition_on(Samples(sec_mat, 3, 0), attr_idx=sint(1), threshold=5,
                                   is_leaf=sint(1))
        runtime_assert_arr_equals(
            [0, 0, 0, 0],
            left.get_active_col(),
            default_test_name()
        )
        runtime_assert_arr_equals(
            [0, 0, 0, 0],
            right.get_active_col(),
            default_test_name()
        )

    def test_determine_if_leaf():
        sec_mat = input_matrix([
            [1, 0, 1],
            [2, 1, 0],
            [3, 0, 0],
            [4, 1, 1]
        ])
        is_leaf, _, _ = determine_if_leaf(Samples(sec_mat, 1))
        runtime_assert_equals(0, is_leaf, default_test_name())

        sec_mat = input_matrix([
            [1, 0, 0],
            [2, 1, 0],
            [3, 0, 0],
            [4, 1, 1]
        ])
        is_leaf, is_dummy, node_class = determine_if_leaf(Samples(sec_mat, 1))
        runtime_assert_arr_equals([1, 0, 1], [is_leaf, is_dummy, node_class], default_test_name())

        sec_mat = input_matrix([
            [1, 0, 1],
            [2, 1, 0],
            [3, 0, 0],
            [4, 1, 0]
        ])
        is_leaf, is_dummy, node_class = determine_if_leaf(Samples(sec_mat, 1))
        runtime_assert_arr_equals([1, 0, 0], [is_leaf, is_dummy, node_class], default_test_name())

        sec_mat = input_matrix([
            [1, 0, 0],
            [2, 1, 0],
            [3, 0, 0],
            [4, 1, 0]
        ])
        is_leaf, is_dummy, _ = determine_if_leaf(Samples(sec_mat, 1))
        runtime_assert_arr_equals([1, 1], [is_leaf, is_dummy], default_test_name())

    def test_while():
        counter = MemValue(sint(5))

        @do_while
        def body():
            counter.write(counter.read() - 1)
            opened = counter.reveal()
            return opened > 1

        runtime_assert_equals(1, sint(counter), default_test_name())

    def test_c45_single_round():
        sec_mat = input_matrix([
            [8, 1, 0, 1],
            [5, 2, 0, 1],
            [7, 3, 1, 1],
            [6, 4, 1, 1]
        ])
        node, left, right = c45_single_round(Samples(sec_mat, 2, 0))
        runtime_assert_equals(1, node.attr_idx, default_test_name())
        runtime_assert_equals(2, node.threshold, default_test_name())
        runtime_assert_arr_equals([1, 1, 0, 0], left.get_active_col(), default_test_name())
        runtime_assert_arr_equals([0, 0, 1, 1], right.get_active_col(), default_test_name())

        sec_mat = input_matrix([
            [1, 0, 1],
            [2, 0, 1],
            [3, 1, 1],
            [4, 1, 1]
        ])
        node, left, right = c45_single_round(Samples(sec_mat, 1, 0))
        runtime_assert_equals(0, node.attr_idx, default_test_name())
        runtime_assert_equals(2, node.threshold, default_test_name())
        runtime_assert_arr_equals([1, 1, 0, 0], left.get_active_col(), default_test_name())
        runtime_assert_arr_equals([0, 0, 1, 1], right.get_active_col(), default_test_name())

    def test_c45():
        sec_mat = input_matrix([
            [8, 1, 1, 1],
            [5, 2, 1, 1],
            [7, 3, 0, 1],
            [6, 4, 0, 1]
        ])
        actual = c45(Samples(sec_mat, 2), max_iteration_count=3)
        expected = \
            DN(1, 2) \
                .l(LN(1)) \
                .r(LN(0))
        runtime_assert_tree_equals(Tree(expected), actual, default_test_name())

        sec_mat = input_matrix([
            [1, 8, 1, 1],
            [2, 9, 1, 1],
            [3, 7, 1, 1],
            [4, 2, 0, 1],
            [5, 1, 1, 1]
        ])
        total_nodes = 2 * (2 ** 2) - 1
        actual = c45(Samples(sec_mat, 2), max_iteration_count=total_nodes)
        expected = \
            DN(1, 2) \
                .l(DN(0, 4)
                   .l(LN(0))
                   .r(LN(1))) \
                .r(LN(1)
                   .l(LN(-1, is_dummy=True))
                   .r(LN(-1, is_dummy=True)))
        runtime_assert_tree_equals(Tree(expected), actual, default_test_name())

        sec_mat = input_matrix([
            [1, 1, 1],
            [2, 1, 1],
            [3, 1, 1],
            [4, 1, 1]
        ])
        actual = c45(Samples(sec_mat, 1), max_iteration_count=3)
        # TODO this makes sense, but is it right?
        # since all samples have the same class,
        # we don't partition on any attribute and end up with a leaf for the root
        expected = \
            LN(1) \
                .l(LN(-1, is_dummy=True)) \
                .r(LN(-1, is_dummy=True))
        runtime_assert_tree_equals(Tree(expected), actual, default_test_name())

    test_argmax()
    test_naive_sort_by()
    test_compute_cont_ginis()
    test_compute_best_gini_cont()
    test_obl_select_col_at()
    test_partition_on()
    test_determine_if_leaf()
    test_while()
    test_c45_single_round()
    test_c45()


def main():
    sec_mat = input_matrix([
        [8, 1, 0, 1],
        [5, 2, 0, 1],
        [7, 3, 1, 1],
        [6, 4, 1, 1]
    ])
    tree = c45(Samples(sec_mat, 2), 3)
    tree.reveal_self()
    tree.print_self()


test()
# main()
