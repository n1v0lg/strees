from Compiler.types import sint, Array, MultiArray, Matrix
from library import for_range_parallel, for_range

try:
    from c45.strees_utils import *
    from c45.c45 import *
    from c45.perm import *
except ImportError:
    pass


# Workaround for working with multiple source files in MP-SPDZ
def super_hacky_import_hack(local_module_names):
    from os.path import dirname, abspath, join
    root_dir = dirname(abspath(program.infile))
    for local_module_name in local_module_names:
        execfile(join(root_dir, local_module_name), globals())


# NOTE: Order matters!
base_dir = "../c45/"
super_hacky_import_hack([
    base_dir + "tree.py",
    base_dir + "strees_utils.py",
    base_dir + "perm.py",
    base_dir + "c45.py"
])


# TODO make sure that things don't get optimized away due to compile time 0 values

def bench_shuffle(num_values):
    """Benchmarks stand-alone shuffle."""
    values = Array(num_values, sint)
    values.assign_all(0)
    config_bits = default_config_shuffle(values)
    default_shuffle(values, config=config_bits, reverse=False)
    print_list(values)


def bench_sort(num_values):
    """Benchmarks stand-alone sort."""
    keys = Array(num_values, sint)
    keys.assign_all(0)

    values = Array(num_values, sint)
    values.assign_all(0)

    default_sort(keys, values)
    print_list(keys)
    print_list(values)


def bench_comp_mat(num_values):
    """Benchmarks naively computing O(n**2) comparison matrix."""
    values = Array(num_values, sint)
    values.assign_all(0)
    comp_mat = Matrix(num_values, num_values, sint)

    @for_range(0, num_values)
    def loop(i):
        @for_range(0, num_values - i)
        def inner(j):
            comp_mat[i][i + j] = values[i] <= values[i + j]


def bench_comp_mat_par(num_values):
    """Benchmarks naively computing O(n**2) comparison matrix."""
    values = Array(num_values, sint)
    values.assign_all(0)
    comp_mat = Matrix(num_values, num_values, sint)
    n_parallel = 32

    @for_range_parallel(n_parallel, num_values)
    def loop(i):
        @for_range_parallel(n_parallel, num_values - i)
        def inner(j):
            comp_mat[i][i + j] = values[i] <= values[i + j]


def bench_argmax_over_fracs(num_values):
    fractions = MultiArray(sizes=[num_values, 3], value_type=sint)
    fractions.assign_all(0)

    res = argmax_over_fracs(fractions)
    print_list(res)


def run_bench():
    args = program.get_args()
    split_args = args[1].split("-")
    operation = split_args[0]
    num_elements = int(split_args[1])
    print "Running %s on %s values." % (operation, num_elements)

    if operation == "shuffle":
        bench_shuffle(num_values=num_elements)
    elif operation == "sort":
        bench_sort(num_values=num_elements)
    elif operation == "comp_mat":
        bench_comp_mat(num_values=num_elements)
    elif operation == "comp_mat_par":
        bench_comp_mat_par(num_values=num_elements)
    elif operation == "argmax":
        bench_argmax_over_fracs(num_values=num_elements)
    else:
        raise Exception("Unknown operation: %s" % operation)


run_bench()
