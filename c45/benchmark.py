try:
    from strees_utils import *
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


def main():
    sec_mat = input_matrix([
        [8, 1, 0, 1],
        [5, 2, 0, 1],
        [7, 3, 1, 1],
        [6, 4, 1, 1]
    ])
    c45(Samples.from_rows(sec_mat, 2), 3) \
        .reveal() \
        .print_self()


main()
