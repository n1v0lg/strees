import sys

from Compiler.types import sint, cint, MemValue, Array
from library import print_ln, if_e, else_, do_while

try:
    from tree import *
    from strees_utils import *
    from perm import *
    from c45 import *
except ImportError:
    pass


# Workaround for working with multiple source files in MP-SPDZ
def super_hacky_import_hack(local_module_names):
    from os.path import dirname, abspath, join
    root_dir = dirname(abspath(program.infile))
    for local_module_name in local_module_names:
        execfile(join(root_dir, local_module_name), globals())


# NOTE: Order matters!
super_hacky_import_hack([
    "tree.py",
    "strees_utils.py",
    "perm.py",
    "c45.py"
])


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
        actual = actual.reveal()
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

    def test_default_sort_store_network():
        sec_mat = input_matrix([
            [7, 0],
            [6, 10],
            [5, 20],
            [4, 30],
            [3, 40],
            [2, 50],
            [1, 60],
            [0, 70]
        ])
        net = default_sort(get_col(sec_mat, 0), get_col(sec_mat, 1), store=True)

        actual_keys = get_col(sec_mat, 0)
        actual_values = get_col(sec_mat, 1)
        # sort is in-place!
        default_sort_from_stored(actual_keys, actual_values, net)

        runtime_assert_arr_equals([0, 1, 2, 3, 4, 5, 6, 7], actual_keys, default_test_name())
        runtime_assert_arr_equals([70, 60, 50, 40, 30, 20, 10, 0], actual_values, default_test_name())

    def test_sort_by():
        sec_mat = input_matrix([
            [2, 0],
            [3, 1],
            [1, 2],
            [0, 3]
        ])
        actual_keys, actual_vals = sort_by(get_col(sec_mat, 0), get_col(sec_mat, 1))

        runtime_assert_arr_equals([0, 1, 2, 3], actual_keys, default_test_name())
        runtime_assert_arr_equals([3, 2, 0, 1], actual_vals, default_test_name())

    def test_compute_cont_ginis():
        sec_mat = input_matrix([
            [3, 0, 1],
            [1, 1, 1],
            [2, 1, 1],
            [4, 0, 1]
        ])
        samples = Samples.from_rows(sec_mat, 1, 0)
        actual = compute_cont_ginis(samples, 0, prep_attributes(samples)[0])
        # TODO double-check these
        expected = [
            [8, alpha_scale(3), 1],
            [16, alpha_scale(4), 2],
            [8, alpha_scale(3), 3],
            [0, alpha_scale(0), 4]
        ]
        runtime_assert_mat_equals(expected, actual, default_test_name())

    def test_compute_best_gini_cont():
        sec_mat = input_matrix([
            [3, 0, 1],
            [1, 1, 1],
            [2, 1, 1],
            [4, 0, 1]
        ])
        samples = Samples.from_rows(sec_mat, 1, 0)
        actual = compute_best_gini_cont(samples, 0, prep_attributes(samples)[0])
        runtime_assert_arr_equals([16, alpha_scale(4), 2], actual, default_test_name())

    def test_obl_select_col_at():
        sec_mat = input_matrix([
            [1, 2, 3, 1, 1],
            [4, 5, 6, 1, 1],
            [7, 8, 9, 1, 1]
        ])
        actual = select_col_at(Samples.from_rows(sec_mat, 3, 0), sint(1))
        runtime_assert_arr_equals([2, 5, 8], actual, default_test_name())

    def test_row_mul():
        sec_mat = input_matrix([
            [1, 2, 3, 4],
            [4, 5, 6, 7],
            [7, 8, 9, 10]
        ])
        eq_flags = input_list([0, 0, 1])
        actual = sint.row_matrix_mul(eq_flags, sec_mat)
        runtime_assert_arr_equals([7, 8, 9, 10], actual, default_test_name())

    def test_partition_on():
        sec_mat = input_matrix([
            [1, 2, 3, 1, 1],
            [4, 5, 6, 1, 1],
            [7, 8, 9, 1, 1],
            [10, 11, 12, 1, 1]
        ])
        left, right = partition_on(Samples.from_rows(sec_mat, 3, 0),
                                   attr_idx=sint(1), threshold=5, is_leaf=sint(0))
        runtime_assert_mat_equals(
            [[1, 2, 3, 1, 1],
             [4, 5, 6, 1, 1],
             [7, 8, 9, 1, 0],
             [10, 11, 12, 1, 0]],
            transpose(left.columns),
            default_test_name()
        )
        runtime_assert_mat_equals(
            [[1, 2, 3, 1, 0],
             [4, 5, 6, 1, 0],
             [7, 8, 9, 1, 1],
             [10, 11, 12, 1, 1]],
            transpose(right.columns),
            default_test_name()
        )

        sec_mat = input_matrix([
            [1, 2, 3, 1, 0],
            [4, 5, 6, 1, 1],
            [7, 8, 9, 1, 0],
            [10, 11, 12, 1, 1]
        ])
        left, right = partition_on(Samples.from_rows(sec_mat, 3, 0), attr_idx=sint(1), threshold=5,
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
        left, right = partition_on(Samples.from_rows(sec_mat, 3, 0), attr_idx=sint(1), threshold=5,
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
        is_leaf, _, _ = determine_if_leaf(Samples.from_rows(sec_mat, 1))
        runtime_assert_equals(0, is_leaf, default_test_name())

        sec_mat = input_matrix([
            [1, 0, 0],
            [2, 1, 0],
            [3, 0, 0],
            [4, 1, 1]
        ])
        is_leaf, is_dummy, node_class = determine_if_leaf(Samples.from_rows(sec_mat, 1))
        runtime_assert_arr_equals([1, 0, 1], [is_leaf, is_dummy, node_class], default_test_name())

        sec_mat = input_matrix([
            [1, 0, 1],
            [2, 1, 0],
            [3, 0, 0],
            [4, 1, 0]
        ])
        is_leaf, is_dummy, node_class = determine_if_leaf(Samples.from_rows(sec_mat, 1))
        runtime_assert_arr_equals([1, 0, 0], [is_leaf, is_dummy, node_class], default_test_name())

        sec_mat = input_matrix([
            [1, 0, 0],
            [2, 1, 0],
            [3, 0, 0],
            [4, 1, 0]
        ])
        is_leaf, is_dummy, _ = determine_if_leaf(Samples.from_rows(sec_mat, 1))
        runtime_assert_arr_equals([1, 1], [is_leaf, is_dummy], default_test_name())

    def test_while():
        num_vals = 5
        counter = MemValue(sint(num_vals - 1))
        source_arr = Array(num_vals, sint)
        for i in range(num_vals):
            source_arr[i] = sint(i)
        target_arr = Array(num_vals, sint)

        @do_while
        def body():
            counter_val = counter.read()
            counter_val_open = counter_val.reveal()
            target_arr[counter_val_open] = source_arr[counter_val_open] + 1
            counter.write(counter_val - 1)
            opened = counter.reveal()
            return opened >= 0

        runtime_assert_arr_equals([1, 2, 3, 4, 5], target_arr, default_test_name())

    def test_reverse_shuffle():
        values = input_list([3, 0, 1, 2, 4, 6, 5, 7])

        config_bits = default_config_shuffle(values)
        default_shuffle(values, config_bits, reverse=False)
        default_shuffle(values, config_bits, reverse=True)
        # expecting original order
        expected = [3, 0, 1, 2, 4, 6, 5, 7]

        runtime_assert_arr_equals(expected, values, default_test_name())

    def test_prep_attr_create():
        sec_mat = input_matrix([
            [3, 0],
            [0, 0],
            [1, 1],
            [2, 1],
            [4, 1],
            [6, 1],
            [5, 0],
            [7, 1]
        ])
        active_col = get_col(sec_mat, 1)
        prep_attr = PermBasedPrepAttribute.create(0, get_col(sec_mat, 0), active_col)
        actual = zip(prep_attr.sorted_val_col, prep_attr.sorted_class_col)
        expected = [
            [0, 0],
            [1, 1],
            [2, 1],
            [3, 0],
            [4, 1],
            [5, 0],
            [6, 1],
            [7, 1]
        ]
        runtime_assert_mat_equals(expected, actual, default_test_name())

        actual = prep_attr.sort(active_col)
        expected = [0, 1, 1, 0, 1, 0, 1, 1]
        runtime_assert_arr_equals(expected, actual, default_test_name())
        # TODO test permutations

    def test_prep_attributes():
        sec_mat = input_matrix([
            [8, 1, 0, 1],
            [5, 2, 0, 1],
            [7, 4, 1, 1],
            [6, 3, 1, 1]
        ])
        prep_attrs = prep_attributes(Samples.from_rows(sec_mat, 2, 0))

        actual = zip(prep_attrs[0].sorted_val_col, prep_attrs[0].sorted_class_col)
        expected = [
            [5, 0],
            [6, 1],
            [7, 1],
            [8, 0]
        ]
        runtime_assert_mat_equals(expected, actual, default_test_name())

        actual = zip(prep_attrs[1].sorted_val_col, prep_attrs[1].sorted_class_col)
        expected = [
            [1, 0],
            [2, 0],
            [3, 1],
            [4, 1]
        ]
        runtime_assert_mat_equals(expected, actual, default_test_name())

    def test_c45_single_round():
        sec_mat = input_matrix([
            [8, 1, 0, 1],
            [5, 2, 0, 1],
            [7, 3, 1, 1],
            [6, 4, 1, 1]
        ])
        samples = Samples.from_rows(sec_mat, 2, 0)
        node, left, right = c45_single_round(samples, prep_attributes(samples))
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
        samples = Samples.from_rows(sec_mat, 1, 0)
        node, left, right = c45_single_round(samples, prep_attributes(samples))
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
        actual = c45(Samples.from_rows(sec_mat, 2), max_tree_depth=2)
        expected = \
            DN(1, 2) \
                .l(LN(1)) \
                .r(LN(0))
        runtime_assert_tree_equals(Tree(expected), actual, default_test_name())

        sec_mat = input_matrix([
            [1, 8, 1, 1],
            [3, 7, 1, 1],
            [4, 2, 0, 1],
            [5, 1, 1, 1]
        ])
        actual = c45(Samples.from_rows(sec_mat, 2), max_tree_depth=3)
        expected = \
            DN(1, 2) \
                .l(DN(1, 1)
                   .l(LN(1))
                   .r(LN(0))) \
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
        actual = c45(Samples.from_rows(sec_mat, 1), max_tree_depth=2)
        # TODO this makes sense, but is it right?
        # since all samples have the same class,
        # we don't partition on any attribute and end up with a leaf for the root
        expected = \
            LN(1) \
                .l(LN(-1, is_dummy=True)) \
                .r(LN(-1, is_dummy=True))
        runtime_assert_tree_equals(Tree(expected), actual, default_test_name())

    test_argmax()
    test_default_sort_store_network()
    test_sort_by()
    test_compute_cont_ginis()
    test_compute_best_gini_cont()
    test_row_mul()
    test_obl_select_col_at()
    test_partition_on()
    test_determine_if_leaf()
    test_while()
    test_reverse_shuffle()
    test_prep_attr_create()
    test_prep_attributes()
    test_c45_single_round()
    test_c45()


test()
