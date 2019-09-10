from Compiler.types import sint, Array
from permutation import rec_shuffle

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
base_dir = "../../c45/"
super_hacky_import_hack([
    base_dir + "tree.py",
    base_dir + "strees_utils.py",
    base_dir + "perm.py",
    base_dir + "c45.py"
])


def gen_dummy_cols(num_rows, num_cols):
    """Generates list of column arrays for given dimensions."""
    cols = [Array(num_rows, sint) for _ in range(num_cols)]
    for col in cols:
        col.assign_all(0)
    return cols


def gen_dummy_samples(num_samples, num_cont_attrs, num_disc_attrs=0):
    """
    Generates Samples with provided dimensions.

    Note: all values are 0. This is fine for benchmarking since the whole algorithm is oblivious.
    """
    columns = gen_dummy_cols(num_samples, num_cont_attrs + num_disc_attrs + 2)
    return Samples(columns, num_cont_attrs, num_disc_attrs)


def bench_shuffle(num_values):
    """Benchmarks stand-alone shuffle."""
    values = Array(num_values, sint)
    values.assign_all(0)
    config_bits = default_config_shuffle(values)
    default_shuffle(values, config=config_bits, reverse=False)
    print_list(values)


def run_bench():
    args = program.get_args()
    operation = args[1]
    num_elements = int(args[2])
    print "Running %s on %s values." % (operation, num_elements)

    if operation == "shuffle":
        bench_shuffle(num_values=num_elements)
    else:
        raise Exception("Unknown operation: %s" % operation)


run_bench()
