from Compiler.types import sint, Array
from permutation import rec_shuffle

try:
    from strees_utils import *
    from c45 import *
    from perm import *
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
    config_bits = rec_config_shuffle(values)
    rec_shuffle(values, config=config_bits, value_type=sint, reverse=False)
    print_list(values)


def bench_prep_attributes(num_samples, num_cont_attrs):
    """Runs attribute pre-processing (sorting and permuting) on sample data with given dimensions."""
    samples = gen_dummy_samples(num_samples, num_cont_attrs)
    prepped = prep_attributes(samples)
    prepped[0].sorted_val_col.reveal()


def bench_c45(num_samples, max_tree_depth, num_cont_attrs, num_disc_attrs=0):
    """Runs c45 algorithm on dummy data with given dimensions."""
    samples = gen_dummy_samples(num_samples, num_cont_attrs, num_disc_attrs)
    c45(samples, max_tree_depth) \
        .reveal() \
        .print_self()


def bench_all():
    bench_shuffle(2048)
    # bench(num_samples=8, max_tree_depth=2, num_cont_attrs=2, num_disc_attrs=0)
    # bench_prep_attributes(num_samples=128, num_cont_attrs=1)
    # bench_c45(num_samples=8, max_tree_depth=2, num_cont_attrs=2, num_disc_attrs=0)
    # bench(num_samples=8, max_tree_depth=4, num_cont_attrs=2, num_disc_attrs=0)


bench_all()
