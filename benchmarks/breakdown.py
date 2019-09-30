from Compiler.types import sint, Array

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


def gen_dummy_cols(num_rows, num_cols):
    """Generates list of column arrays for given dimensions."""
    cols = [Array(num_rows, sint) for _ in range(num_cols)]
    for col in cols:
        col.assign_all(1)
    return cols


def gen_dummy_samples(num_samples, num_cont_attrs, num_disc_attrs=0):
    """
    Generates Samples with provided dimensions.

    Note: all values are 0. This is fine for benchmarking since the whole algorithm is oblivious.
    """
    columns = gen_dummy_cols(num_samples, num_cont_attrs + num_disc_attrs + 2)
    return Samples(columns, num_cont_attrs, num_disc_attrs)


def bench_prep_attributes(num_samples, num_cont_attrs):
    """Runs attribute pre-processing (sorting and permuting) on sample data with given dimensions."""
    samples = gen_dummy_samples(num_samples, num_cont_attrs)
    prep_attributes(samples)


def bench_prep_attr_sort(num_samples, preprocessor):
    """Runs PrepAttr.sort on given number of samples."""
    samples = gen_dummy_samples(num_samples, 1)
    prepped = prep_attributes(samples, preprocessor=preprocessor)
    prepped[0].sort(samples.get_active_col())


def bench_c45_single_round(num_samples, num_cont_attrs, preprocessor, num_disc_attrs=0):
    """Runs c45 algorithm on dummy data with given dimensions."""
    samples = gen_dummy_samples(num_samples, num_cont_attrs, num_disc_attrs)
    prepped = prep_attributes(samples, preprocessor=preprocessor)
    c45_single_round(samples, prepped)


def bench_select_col(num_samples, num_cont_attrs):
    samples = gen_dummy_samples(num_samples, num_cont_attrs, 0)
    select_col_at(samples, sint(1))


def bench_compute_ginis(num_samples, num_cont_attrs=2):
    samples = gen_dummy_samples(num_samples, num_cont_attrs, 0)
    prepped = prep_attributes(samples, PermBasedPrepAttribute.create_dummy)
    compute_cont_ginis(samples, 0, prepped[0])


def run_bench():
    args = program.get_args()
    split_args = args[1].split("-")
    operation = split_args[0]
    num_elements = int(split_args[1])
    num_cont_attrs = int(split_args[2])
    print "Running %s on %s with %s cont attrs." \
          % (operation, num_elements, num_cont_attrs)

    if operation == "prep":
        bench_prep_attributes(num_samples=num_elements, num_cont_attrs=num_cont_attrs)
    elif operation == "select_col":
        bench_select_col(num_samples=num_elements, num_cont_attrs=num_cont_attrs)
    elif operation == "dummy_perm_sort":
        bench_prep_attr_sort(num_samples=num_elements, preprocessor=PermBasedPrepAttribute.create_dummy)
    elif operation == "dummy_sort_sort":
        bench_prep_attr_sort(num_samples=num_elements, preprocessor=SortNetBasedPrepAttribute.create_dummy)
    elif operation == "single_perm_dummy":
        bench_c45_single_round(num_samples=num_elements, preprocessor=PermBasedPrepAttribute.create_dummy,
                               num_cont_attrs=num_cont_attrs)
    elif operation == "single_perm_both":
        bench_c45_single_round(num_samples=num_elements, preprocessor=PermBasedPrepAttribute.create,
                               num_cont_attrs=num_cont_attrs)
    elif operation == "single_sort_dummy":
        bench_c45_single_round(num_samples=num_elements, preprocessor=SortNetBasedPrepAttribute.create_dummy,
                               num_cont_attrs=num_cont_attrs)
    elif operation == "compute_ginis":
        bench_compute_ginis(num_samples=num_elements)
    else:
        raise Exception("Unknown operation: %s" % operation)


run_bench()
