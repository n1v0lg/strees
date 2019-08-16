import math

from Compiler.types import sint, cint, MemValue, Array
from library import for_range_parallel, for_range, function_block
from permutation import configure_waksman, config_shuffle

# Make IDE happy
try:
    from strees_utils import *
except Exception:
    pass

# NOTE most of these methods are taken from MP-SPDZ and updated to account for shuffling rows and fixing some bugs(?)
WAKSMAN_FUNCTIONS = {}
inwards = 1


def cond_swap_bit(x, y, b):
    """ swap if b == 1 """
    if x is None:
        return y, None
    elif y is None:
        return x, None
    if isinstance(x, list):
        t = [(xi - yi) * b for xi, yi in zip(x, y)]
        return [xi - ti for xi, ti in zip(x, t)], \
               [yi + ti for yi, ti in zip(y, t)]
    else:
        t = (x - y) * b
        return x - t, y + t


# NOTE: taken directly from MP-SPDZ, with some fixes/modifications in place

def config_shuffle_with_perm(perm, value_type):
    """ Compute config for oblivious shuffling.

    Take mod 2 for active sec. """
    n = len(perm)
    if n & (n - 1) != 0:
        # pad permutation to power of 2
        m = 2 ** int(math.ceil(math.log(n, 2)))
        perm += range(n, m)
    config_bits = configure_waksman(perm)
    # 2-D array
    config = Array(len(config_bits) * len(perm), value_type.reg_type)
    for i, c in enumerate(config_bits):
        for j, b in enumerate(c):
            config[i * len(perm) + j] = b
    return config


def fixed_iter_waksman(a, config, reverse=False):
    """ Iterative Waksman algorithm, compilable for large inputs. Input
    must be an Array. """
    n = len(a)
    # if not isinstance(a, Array):
    #    raise CompilerError('Input must be an Array')
    depth = MemValue(0)
    nblocks = MemValue(1)
    size = MemValue(0)
    a2 = Array(n, a[0].reg_type)

    # config_array = Array(n, a[0].reg_type)
    # reverse = (int(reverse))

    def create_round_fn(n, reg_type):
        if (n, reg_type, inwards) in WAKSMAN_FUNCTIONS:
            return WAKSMAN_FUNCTIONS[(n, reg_type, inwards)]

        def do_round(size, config_address, a_address, a2_address, inwards):
            A = Array(n, reg_type, a_address)
            A2 = Array(n, reg_type, a2_address)
            C = Array(n, reg_type, config_address)
            inwards = MemValue(inwards)
            outwards = MemValue(1 - inwards)

            sizeval = size

            # for k in range(n/2):
            @for_range_parallel(200, n / 2)
            def f(k):
                j = cint(k) % sizeval
                i = (cint(k) - j) / sizeval
                base = 2 * i * sizeval

                in1, in2 = (base + j + j * inwards), (base + j + j * inwards + 1 * inwards + sizeval * outwards)
                out1, out2 = (base + j + j * outwards), (base + j + j * outwards + 1 * outwards + sizeval * inwards)

                if inwards:
                    if reverse:
                        c = C[base + j + sizeval]
                    else:
                        c = C[base + j]
                else:
                    if reverse:
                        c = C[base + j]
                    else:
                        c = C[base + j + sizeval]
                A2[out1], A2[out2] = cond_swap_bit(A[in1], A[in2], c)

        fn = function_block(do_round)
        WAKSMAN_FUNCTIONS[(n, reg_type, inwards)] = fn
        return fn

    do_round = create_round_fn(n, a[0].reg_type)

    logn = int(math.log(n, 2))

    # going into middle of network
    @for_range(logn)
    def f(i):
        size.write(n / (2 * nblocks))
        conf_address = MemValue(config.address + depth.read() * n)
        do_round(size, conf_address, a.address, a2.address, cint(1))

        for i in range(n):
            a[i] = a2[i]

        nblocks.write(nblocks * 2)
        depth.write(depth + 1)

    nblocks.write(nblocks / 4)
    depth.write(depth - 2)

    # and back out
    @for_range(logn - 1)
    def f(i):
        size.write(n / (2 * nblocks))
        conf_address = MemValue(config.address + depth.read() * n)
        do_round(size, conf_address, a.address, a2.address, cint(0))

        for i in range(n):
            a[i] = a2[i]

        nblocks.write(nblocks / 2)
        depth.write(depth - 1)


def fixed_shuffle(x, config=None, value_type=sint, reverse=False):
    """ Simulate secure shuffling with Waksman network for 2 players.
    WARNING: This is not a properly secure implementation but has roughly the right complexity.

    Returns the network switching config so it may be re-used later.  """
    n = len(x)
    m = 2 ** int(math.ceil(math.log(n, 2)))
    if config is None:
        config = config_shuffle(n, value_type)

    if isinstance(x, list):
        if isinstance(x[0], list):
            length = len(x[0])
            for i in range(length):
                xi = Array(m, value_type.reg_type)
                for j in range(n):
                    xi[j] = x[j][i]
                for j in range(n, m):
                    xi[j] = value_type(0)
                fixed_iter_waksman(xi, config, reverse=reverse)
                fixed_iter_waksman(xi, config, reverse=reverse)
        else:
            xa = Array(m, value_type.reg_type)
            for i in range(n):
                xa[i] = x[i]
            for i in range(n, m):
                xa[i] = value_type(0)
            fixed_iter_waksman(xa, config, reverse=reverse)
            fixed_iter_waksman(xa, config, reverse=reverse)
    elif isinstance(x, Array):
        if len(x) != m and config is None:
            raise Exception('Non-power of 2 Array input not yet supported')
        fixed_iter_waksman(x, config, reverse=reverse)
        fixed_iter_waksman(x, config, reverse=reverse)
    else:
        raise Exception('Invalid type for shuffle:', type(x))

    return config


def sort_and_permute(rows, attr_idx):
    """Sorts and permutes rows according to specified permutation."""
    config_bits = config_shuffle(len(rows), value_type=sint)
    sorted_rows = naive_sort_by(rows, attr_idx)
    fixed_shuffle(sorted_rows, config=config_bits)

    return sorted_rows, config_bits
