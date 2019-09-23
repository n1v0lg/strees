#!/usr/bin/env python

from Compiler.types import sint, Array, Matrix
from library import print_ln, if_, for_range, for_range_parallel

# Make IDE happy
try:
    from tree import *
    from strees_utils import *
    from perm import *
except ImportError:
    pass


class Samples:

    def __init__(self, columns, n, m=0):
        """Create Samples object.

        :param columns: raw matrix containing continuous and discrete (binary) attribute values, category (binary),
        and indicator vector, in column format
        :param n: number of continuous attributes
        :param m: number of discrete attributes
        """
        total_cols = n + m + 2
        if len(columns) != total_cols:
            raise Exception("Wrong number of cols. in samples matrix")
        self.n = n
        self.m = m
        self.columns = columns

    @staticmethod
    def from_rows(rows, n, m=0):
        return Samples(to_col_arrays(rows), n, m)

    def get_col(self, col_idx):
        return self.columns[col_idx]

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
        if len(updated_actives) != len(self):
            raise Exception("Incorrect number of values")
        new_columns = [col for col in self.columns[:-1]]
        new_columns.append(updated_actives)
        return Samples(new_columns, self.n, self.m)

    def __len__(self):
        return len(self.get_active_col())


class PrepAttribute:

    def __init__(self, attr_idx, sorted_val_col, sorted_class_col):
        """Create PrepAttribute object.

        :param attr_idx:
        :param sorted_val_col:
        :param sorted_class_col:
        """
        self.attr_idx = attr_idx
        self.sorted_val_col = sorted_val_col
        self.sorted_class_col = sorted_class_col


class PermBasedPrepAttribute(PrepAttribute):

    def __init__(self, attr_idx, sorted_val_col, sorted_class_col, rand_perm, open_perm):
        """Create PermBasedPrepAttribute object.

        :param attr_idx:
        :param sorted_val_col:
        :param sorted_class_col:
        :param rand_perm:
        :param open_perm:
        """
        PrepAttribute.__init__(self, attr_idx, sorted_val_col, sorted_class_col)
        self.rand_perm = rand_perm
        self.open_perm = open_perm

    @staticmethod
    def create(attr_idx, val_col, class_col):
        """Creates PermBasedPrepAttribute for given attribute column.

        :param attr_idx:
        :param val_col:
        :param class_col:
        :return:
        """
        n = len(val_col)
        indexes = get_indexes(n)
        sorted_val_col, order_col, rand_perm = sort_and_permute(val_col, indexes)
        open_perm = order_col.reveal()
        sorted_class_col = PermBasedPrepAttribute._sort(class_col, rand_perm, open_perm)
        return PermBasedPrepAttribute(attr_idx, sorted_val_col, sorted_class_col, rand_perm, open_perm)

    @staticmethod
    def create_dummy(attr_idx, val_col, class_col):
        """Creates dummy attribute, WITHOUT running a sort.

        NOTE for benchmarking only.
        """
        n = len(val_col)
        open_perm = get_indexes(n).reveal()
        # null permutation
        rand_perm = config_shuffle_given_perm(list(range(n)))

        return PermBasedPrepAttribute(attr_idx, val_col, class_col, rand_perm, open_perm)

    @staticmethod
    def _sort(col, rand_perm, open_perm):
        """Sorts given column using random perm and open perm."""
        sorted_col = open_permute(col, open_perm)
        default_shuffle(sorted_col, config=rand_perm, reverse=True)
        return sorted_col

    def sort(self, col):
        return self._sort(col, self.rand_perm, self.open_perm)


class SortNetBasedPrepAttribute(PrepAttribute):

    def __init__(self, attr_idx, sorted_val_col, sorted_class_col, sorting_net):
        """Create SortNetBasedPrepAttribute object.

        :param attr_idx:
        :param sorted_val_col:
        :param sorted_class_col:
        :param sorting_net:
        """
        PrepAttribute.__init__(self, attr_idx, sorted_val_col, sorted_class_col)
        self.sorting_net = sorting_net

    @staticmethod
    def create(attr_idx, val_col, class_col):
        """Creates PrepAttribute for given attribute column.

        :param attr_idx:
        :param val_col:
        :param class_col:
        :return:
        """
        sorted_val_col, sorted_class_col, sorting_net = sort_by(val_col, class_col, store=True)
        return SortNetBasedPrepAttribute(attr_idx, sorted_val_col, sorted_class_col, sorting_net)

    @staticmethod
    def create_dummy(attr_idx, val_col, class_col):
        """Creates dummy attribute, WITHOUT running a sort.

        NOTE for benchmarking only.
        """
        sorting_net = gen_dummy_net(len(val_col))
        return SortNetBasedPrepAttribute(attr_idx, val_col, class_col, sorting_net)

    def sort(self, col):
        """Sorts given column according to stored sorting network."""
        res = col[:]
        default_sort_from_stored(res, self.sorting_net)
        return res


def argmax_over_fracs(elements):
    """Computes argmax of elements.

    Supports arbitrary number of columns but assumes that numerator and denominator are the first two columns.

    NOTE: undefined if denominator of any fraction is 0.

    :param elements: represented as rows of form [numerator, denominator, ...]
    :return row with max fraction
    """

    def _select_larger(row_a, row_b):
        num_a, denom_a = row_a[0], row_a[1]
        num_b, denom_b = row_b[0], row_b[1]

        a_gt = (num_a * denom_b) > (num_b * denom_a)
        return if_else_row(a_gt, row_a, row_b)

    n_elements = len(elements)

    if n_elements == 0:
        raise Exception("No elements to argmax on")
    if n_elements == 1:
        return elements[0]

    if not is_two_pow(n_elements):
        raise Exception("Only support powers of two")

    step_size = 1

    while n_elements > step_size:
        num_its = (n_elements // step_size) - 1

        # @for_range(0, num_its)
        @for_range_parallel(min(32, num_its), num_its)
        def _(i):
            left_idx = i * step_size
            left = elements[left_idx]

            right_idx = (i + 1) * step_size
            right = elements[right_idx]

            elements[i * step_size][:] = _select_larger(left, right)

        step_size *= 2

    return elements[0]


def compute_cont_ginis(samples, attr_col_idx, prep_attr):
    """Computes gini values as fractions for given attribute.

    :param samples:
    :param attr_col_idx:
    :param prep_attr:
    :return:
    """

    if not samples.is_cont_attribute(attr_col_idx):
        raise Exception("Can only call this on continuous attribute")

    val_col = prep_attr.sorted_val_col
    class_col = prep_attr.sorted_class_col

    # put active col into order induced by this attribute
    is_active = prep_attr.sort(samples.get_active_col())

    # all samples of class 1 that are still active
    is_one = pairwise_and(class_col, is_active)
    # active 0 samples
    is_zero = pairwise_and(neg(is_one), is_active)

    num_samples = len(val_col)
    fractions = Matrix(num_samples, 3, value_type=sint)

    leq_this = Acc(sint(0))
    gt_this = Acc(iter_sum(is_active))

    ones_leq = Acc(sint(0))
    ones_gt = Acc(iter_sum(is_one))

    zeroes_leq = Acc(sint(0))
    zeroes_gt = Acc(iter_sum(is_zero))

    @for_range(0, num_samples - 1)
    def _(row_idx):
        threshold = val_col[row_idx]
        leq_this.inc_by(is_active[row_idx])
        gt_this.dec_by(is_active[row_idx])

        ones_leq.inc_by(is_one[row_idx])
        ones_gt.dec_by(is_one[row_idx])

        zeroes_leq.inc_by(is_zero[row_idx])
        zeroes_gt.dec_by(is_zero[row_idx])

        # TODO pull this out into separate, parallel loop
        numerator, denominator = _compute_gini_fraction(
            leq_this.get_val(), gt_this.get_val(),
            ones_leq.get_val(), ones_gt.get_val(),
            zeroes_leq.get_val(), zeroes_gt.get_val()
        )
        denominator = alpha_scale(denominator)
        fractions[row_idx][0] = numerator
        fractions[row_idx][1] = denominator
        fractions[row_idx][2] = threshold

    # TODO this can go away once the alpha fix is implemented
    # include fraction for splitting on last term
    last_threshold = val_col[-1]
    fractions[num_samples - 1][0] = sint(0)
    fractions[num_samples - 1][1] = alpha_scale(sint(0))
    fractions[num_samples - 1][2] = last_threshold

    is_last_active = compute_is_last_active(val_col, is_active)

    # Zero out numerator for all entries that are repeated but not last in sequence
    # since these are not valid splitting points
    @for_range_parallel(min(32, len(val_col)), len(val_col))
    def _(i):
        fractions[i][0] *= is_last_active[i]

    return fractions


def _compute_gini_fraction(leq_this, gt_this, ones_leq, ones_gt, zeroes_leq, zeroes_gt):
    # Note that ones_leq = |D'_{C_{attr_col_idx} <= c_{attr_col_idx, row_idx}} ^ D'_{Y = 1}|
    # where D' is D sorted by the attribute at attr_col_idx
    numerator_one_term = \
        (ones_leq ** 2) * gt_this + (ones_gt ** 2) * leq_this
    numerator_zero_term = \
        (zeroes_leq ** 2) * gt_this + (zeroes_gt ** 2) * leq_this
    numerator = numerator_one_term + numerator_zero_term
    denominator = leq_this * gt_this
    return numerator, denominator


def compute_best_gini_cont(samples, attr_col_idx, prep_attr):
    """Computes best gini for given attribute.

    :param samples:
    :param attr_col_idx:
    :param prep_attr:
    :return:
    """
    cand_ginis = compute_cont_ginis(samples, attr_col_idx, prep_attr)
    return argmax_over_fracs(cand_ginis)


def select_col_at(samples, idx):
    """Selects column at given index, obliviously if index is secret.

    :param samples:
    :param idx:
    :return:
    """

    @debug_only
    def debug_sanity_check(col_idx):
        # INSECURE for debugging only!
        # Verifies that idx is within range
        @if_((col_idx >= samples.n + samples.m).reveal())
        def _():
            print_ln("%s index is out of range.", MPC_ERROR_FLAG)

    if isinstance(idx, int):
        return samples.get_col(idx)

    if not isinstance(idx, sint):
        raise Exception("Only use this if index is secret or int")

    debug_sanity_check(idx)
    # TODO for_range
    eq_flags = Array.create_from(idx == i for i in range(samples.n + samples.m))
    selected = sint.row_matrix_mul(eq_flags, samples.columns[0:samples.n + samples.m])

    return Array.create_from(v for v in selected)


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

    # TODO this only works for binary discrete attributes
    go_left = lt_threshold(selected_col, threshold)
    go_right = neg(go_left)

    # mask out rows that are already inactive
    active_col = samples.get_active_col()
    go_left = pairwise_and(go_left, active_col)
    # TODO can we derive this from go_left?
    go_right = pairwise_and(go_right, active_col)

    # set both indicator vectors to 0 if we're at a leaf node
    is_internal = neg(is_leaf)
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

    active_ones = pairwise_and(is_category_one, active_col)
    active_zeroes = pairwise_and(is_category_zero, active_col)

    total_actives = iter_sum(active_col)

    all_inactive = total_actives == 0
    all_ones = iter_sum(active_ones) == total_actives
    all_zeroes = iter_sum(active_zeroes) == total_actives

    is_leaf = log_or(all_inactive, log_or(all_ones, all_zeroes))
    return is_leaf, all_inactive, all_ones


def c45_single_round(samples, prep_attrs):
    """Runs single round of C4.5 algorithm.

    :param samples:
    :param prep_attrs:
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
    candidates = Matrix(samples.n, 4, value_type=sint)

    for c in range(samples.n):
        program.curr_tape.start_new_basicblock()
        num, denom, th = compute_best_gini_cont(samples, c, prep_attrs[c])
        candidates[c][0] = num
        candidates[c][1] = denom
        candidates[c][2] = c
        candidates[c][3] = th

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


def prep_attributes(samples, preprocessor=PermBasedPrepAttribute.create):
    """Pre-processes sorts for all continuous attributes.

    :param samples:
    :param preprocessor:
    :return:
    """
    class_col = samples.get_class_col()
    prepped = []
    for attr_idx in range(samples.n):
        program.curr_tape.start_new_basicblock()
        prepped.append(preprocessor(attr_idx, samples.get_col(attr_idx), class_col))
    return prepped


def _c45(input_samples, max_iteration_count, prep_attrs=None):
    """Runs C4.5 algorithm to construct decision tree.

    This implementation uses an iterative approach as opposed to the more obvious recursive approach since this seems
    better aligned with MP-SPDZ.

    :param input_samples:
    :param max_iteration_count: upper limit on iterations to generate decision tree for samples
    :param prep_attrs:
    :return:
    """
    if prep_attrs is None:
        prep_attrs = prep_attributes(input_samples)

    queue = deque([(None, input_samples)])
    root = None

    # Create tree in a BFS-like traversal
    for _ in range(max_iteration_count):
        # TODO this fixes the open instruction merge issue
        program.curr_tape.start_new_basicblock()
        parent, samples = queue.popleft()
        node, left_samples, right_samples = c45_single_round(samples, prep_attrs)
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

    return Tree(root)


def c45(input_samples, max_tree_depth, prep_attrs=None):
    """Runs C4.5 algorithm to construct decision tree.

    :param input_samples:
    :param max_tree_depth: depth of resulting tree (may include dummy nodes)
    :param prep_attrs:
    :return:"""
    max_iteration_count = (2 ** max_tree_depth) - 1
    return _c45(input_samples, max_iteration_count, prep_attrs)
