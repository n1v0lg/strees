#!/usr/bin/env python

from collections import deque

from library import print_ln, print_str, if_e, else_


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


# TODO get rid of side-effects

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
