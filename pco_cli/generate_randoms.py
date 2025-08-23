# python 3
# pip requirement: numpy

import argparse
import os
from datetime import datetime
from pathlib import Path

import numpy as np

n = 1_000_000
max_n = 2**24 - 1


def write_generic(strs, arr, full_name, base_dir):
    print(f"writing {full_name}...")
    joined = "\n".join(strs)
    with open(base_dir / "txt" / f"{full_name}.txt", "w") as f:
        f.write(joined)
    with open(base_dir / "binary" / f"{full_name}.bin", "wb") as f:
        f.write(arr.tobytes())


WRITERS = {}


def writer(f):
    name = f.__name__
    assert name.startswith("write_")
    dtype = name.removeprefix("write_")
    WRITERS[dtype] = f
    return f


def write_dispatch(dtype, arr, name, base_dir):
    WRITERS[dtype](arr, name, base_dir)


@writer
def write_i32(arr, name, base_dir):
    if arr.dtype != np.int32:
        arr = np.floor(arr).astype(np.int32)
    strs = [str(x) for x in arr]
    full_name = f"i32_{name}"
    write_generic(strs, arr, full_name, base_dir)


@writer
def write_u32(arr, name, base_dir):
    if arr.dtype != np.uint32:
        arr = np.floor(arr).astype(np.uint32)
    strs = [str(x) for x in arr]
    full_name = f"u32_{name}"
    write_generic(strs, arr, full_name, base_dir)


@writer
def write_i64(arr, name, base_dir):
    if arr.dtype != np.int64:
        arr = np.floor(arr).astype(np.int64)
    strs = [str(x) for x in arr]
    full_name = f"i64_{name}"
    write_generic(strs, arr, full_name, base_dir)


@writer
def write_f16(arr, name, base_dir):
    arr = arr.astype(np.float16)
    strs = [str(x) for x in arr]
    full_name = f"f16_{name}"
    write_generic(strs, arr, full_name, base_dir)


@writer
def write_f32(arr, name, base_dir):
    arr = arr.astype(np.float32)
    strs = [str(x) for x in arr]
    full_name = f"f32_{name}"
    write_generic(strs, arr, full_name, base_dir)


@writer
def write_f64(arr, name, base_dir):
    arr = arr.astype(np.float64)
    strs = [str(x) for x in arr]
    full_name = f"f64_{name}"
    write_generic(strs, arr, full_name, base_dir)


@writer
def write_timestamp_micros(arr, name, base_dir):
    if arr.dtype != np.int64:
        arr = np.floor(arr).astype(np.int64)
    ts = [datetime.utcfromtimestamp(x / 10**6) for x in arr]
    strs = [x.strftime("%Y-%m-%dT%H:%M:%S:%fZ") for x in ts]
    full_name = f"micros_{name}"
    write_generic(strs, arr, full_name, base_dir)


DATA_GENS = {}


def datagen(*dtypes):
    def decorate(f):
        name = f.__name__
        DATA_GENS[name] = (f, dtypes)
        return f

    return decorate


@datagen("i64")
def geo():
    return np.random.geometric(p=0.001, size=n)


def fixed_median_lomax(a, median):
    unscaled_median = 2 ** (1 / a) - 1
    return np.random.pareto(a=a, size=n) / unscaled_median * median


@datagen("i32", "u32", "i64")
def lomax05():
    return fixed_median_lomax(0.5, 1000)


@datagen("i64")
def uniform():
    return np.random.randint(-(2**63), 2**63, size=max_n, dtype=np.int64)[:n]


# disable the following by default because it's kinda a waste of disk:
# @datagen('i64')
# def uniform_xl():
#   return np.random.randint(-2**63, 2**63, size=max_n)


@datagen("i64")
def constant():
    return np.repeat(77777, n)


@datagen("i64")
def sparse():
    return np.random.binomial(1, p=0.01, size=n)


money = {}


def gen_money_once():
    if "dollars" in money:
        return
    dollars = np.floor(fixed_median_lomax(1.5, 5)).astype(np.int64)
    cents = np.random.randint(0, 100, size=n)
    p = np.random.uniform(size=n)
    cents[p < 0.9] = 99
    cents[p < 0.75] = 98
    cents[p < 0.6] = 95
    cents[p < 0.45] = 75
    cents[p < 0.4] = 50
    cents[p < 0.25] = 25
    cents[p < 0.15] = 0
    total_cents = dollars * 100 + cents

    money["dollars"] = dollars
    money["cents"] = cents
    money["total_cents"] = total_cents


@datagen("i64")
def dollars():
    gen_money_once()
    return money["dollars"]


@datagen("i64")
def cents():
    gen_money_once()
    return money["cents"]


@datagen("i64")
def total_cents():
    gen_money_once()
    return money["total_cents"]


@datagen("i64", "f64")
def slow_cosine():
    periods = 103  # something prime
    period = n / periods
    return 100_000 * np.cos(np.arange(n) * 2 * np.pi / period)


# Including lower precisions mostly just to test performance bottlenecks
@datagen("f64", "f32", "f16")
def normal():
    return np.random.normal(size=n)


@datagen("f32")
def log_normal():
    return np.exp(normal())


@datagen("f32")
def csum():
    return np.cumsum(log_normal() - np.exp(0.5))


# timestamps increasing 1s at a time on average from 2022-01-01T00:00:00 with
# 1s random jitter
@datagen("timestamp_micros")
def near_linear():
    return 10**6 * (1640995200 + np.arange(n) + np.random.normal(size=n))


# millisecond timestamps compressed as microseconds
@datagen("timestamp_micros")
def millis():
    return 10**3 * (1640995200000 + np.random.randint(0, 10**9, size=n, dtype=np.int64))


# integers compressed as floats
@datagen("f64")
def integers():
    return np.random.randint(0, 2**30, size=n)


# `float32`s compressed as `float64`s
@datagen("f64")
def quantized_normal():
    return np.random.normal(size=n).astype(np.float32).astype(np.float64)


# decimal floats
@datagen("f64")
def decimal():
    return np.random.randint(1000, 10000, size=n) / 100


@datagen("f64")
def radians():
    return (np.arange(n) + 10) * np.pi


# 10 interleaved 0th order sequences with different scales
@datagen("i64")
def interl0():
    bases = 10 ** np.arange(10)
    interleaved = bases[None, :] + np.random.normal(scale=22, size=[n // 10, 10])
    return interleaved.reshape(-1)


# 10 interleaved 1st order sequences with different scales
def interl1_helper():
    deltas = np.random.randint(-10, 10, size=[n // 10, 10])
    bases = 10 ** np.arange(10)
    return bases[None, :] + np.cumsum(deltas, axis=0)


@datagen("i64")
def interl1():
    interleaved = interl1_helper()
    return interleaved.reshape(-1)


# the same as interleaved, but shuffled within each group of 10
@datagen("i64")
def interl_scrambl1():
    interleaved = interl1_helper()
    idxs = np.random.rand(*interleaved.shape).argsort(axis=1)
    interleaved_scrambled = np.take_along_axis(interleaved, idxs, axis=1)
    return interleaved_scrambled.reshape(-1)


# a sequence whose variance gradually increases
@datagen("i64")
def dist_shift():
    log_std = np.linspace(-1.5, 25, n)
    return 0.5 + np.exp(log_std) * np.random.normal(size=n)


# a diabolically hard sequence
# * float decimals plus small jitter
# * numerous interleaved subsequences with occasionally missing elements
# * each subsequence benefits from delta
# * distribution shift on delta size
@datagen("f64")
def diablo():
    n_subseqs = 77
    subseq_vals = np.random.randint(1e4, 1e5, size=n_subseqs)
    the_data = []
    log_delta_scale = 0.0
    frequency = 1e-4
    add_scale = np.sqrt(np.exp(2 * frequency) - 1)
    mult = np.exp(-frequency)
    while len(the_data) < n:
        the_data.extend(subseq_vals[np.random.uniform(size=n_subseqs) > 0.9])
        log_delta_scale += np.random.normal() * add_scale
        log_delta_scale *= mult
        delta_scale = 3 * np.exp(log_delta_scale)
        subseq_vals += np.random.uniform(
            -delta_scale, delta_scale, size=n_subseqs
        ).astype(int)
    the_data = np.array(the_data[:n]).astype(np.float64)
    the_data /= 100.0
    machine_eps = 1.0e-52
    the_data *= np.random.uniform(1 - 2 * machine_eps, 1 + 3 * machine_eps, size=n)
    return the_data


def uniquify_preserving_order(xs):
    return list(dict.fromkeys(xs))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--base_dir",
        type=Path,
        default=Path("data"),
        help="Directory in which to write the output data",
    )
    parser.add_argument(
        "datasets",
        type=str,
        nargs="*",
        help=(
            'Datasets to generate. If not provided, all'
            ' datasets are generated. Available datasets are:'
            f' {", ".join(DATA_GENS)}'
        ),
    )
    args = parser.parse_args()

    os.makedirs(args.base_dir / "txt", exist_ok=True)
    os.makedirs(args.base_dir / "binary", exist_ok=True)

    if not args.datasets:
        datasets = list(DATA_GENS)
    else:
        datasets = uniquify_preserving_order(args.datasets)
        for name in datasets:
            if name not in DATA_GENS:
                raise NotImplementedError(f"Unrecognized dataset name: {name}")

    for name in datasets:
        (f, dtypes) = DATA_GENS[name]
        np.random.seed(0)
        data = f()
        for dtype in dtypes:
            write_dispatch(dtype, data, name, args.base_dir)
