<p align="center">
  <img
    alt="Pco logo: a pico-scale, compressed version of the Pyramid of Khafre in the palm of your hand" src="images/logo.svg"
    width="160px"
  >
</p>

[![crates.io][crates-badge]][crates-url]
[![pypi.org][pypi-badge]][pypi-url]
[![https://central.sonatype.com/][maven-central-badge]][maven-central-url]

[crates-badge]: https://img.shields.io/crates/v/pco.svg

[crates-url]: https://crates.io/crates/pco

[pypi-badge]: https://img.shields.io/pypi/v/pcodec.svg

[pypi-url]: https://pypi.org/project/pcodec/

[maven-central-badge]: https://img.shields.io/maven-central/v/io.github.pcodec/pco-jni

[maven-central-url]: https://central.sonatype.com/artifact/io.github.pcodec/pco-jni

# Pcodec

<p align="center">
  <img
    alt="bar charts showing better compression for Pco than zstd parquet or blosc"
    src="images/real_world_compression_ratio.svg"
    width="700px"
  >
</p>

Pcodec (or Pco) losslessly compresses and decompresses
numerical sequences with
[high compression ratio and moderately fast speed](docs/benchmark_results.md).

**Use cases include:**

* columnar data
* long-term time series data
* serving numerical data to web clients
* low-bandwidth communication

**Data types:**
`u16`, `u32`, `u64`, `i16`, `i32`, `i64`, `f16`, `f32`, `f64`

## Get Started

[Use the CLI](./pco_cli/README.md) (also supports benchmarking)

[Use the Rust API](https://docs.rs/pco/latest/pco/)

[Use the Python API](./pco_python/README.md)

[Use the Java API](./pco_java/README.md)

## How is Pco so much better than alternatives?

Pco is designed specifically for numerical data, whereas alternatives rely on
general-purpose (LZ) compressors that target string or binary data.
Pco uses a holistic, 3-step approach:

* **modes**.
  Pco identifies an approximate structure of the numbers called a
  mode and then uses it to split numbers into "latents".
  As an example, if all numbers are approximately multiples of 777, int mult mode
  splits each number `x` into latent variables `l_0` and
  `l_1` such that `x = 777 * l_0 + l_1`.
  Most natural data uses classic mode, which simply matches `x = l_0`.
* **delta encoding**.
  Pco identifies whether certain latent variables would be better compressed as
  deltas between consecutive elements (or deltas of deltas, or deltas with 
  lookback).
  If so, it takes differences.
* **binning**.
  This is the heart and most novel part of Pco.
  Pco represents each (delta-encoded) latent variable as an approximate,
  entropy-coded bin paired an exact offset into that bin.
  This nears the Shannon entropy of any smooth distribution very efficiently.

These 3 steps cohesively capture most entropy of numerical data without waste.

In contrast, LZ compressors are only effective for patterns like repeating
exact sequences of numbers.
Such patterns constitute just a small fraction of most numerical data's
entropy.

## Usage Details

### Wrapped or Standalone

Pco is designed to embed into wrapping formats.
It provides a powerful wrapped API with the building blocks to interleave it
with the wrapping format.
This is useful if the wrapping format needs to support things like nullability,
multiple columns, random access, or seeking.

The standalone format is a minimal implementation of a wrapped format.
It supports batched decompression only with no other niceties.
It is mainly recommended for quick proofs of concept and benchmarking.

### Granularity

Pco has a hierarchy of multiple batches per page; multiple pages per chunk; and
multiple chunks per file.
By default Pco uses up to 2^18 (~262k) numbers per chunk if available.

|       | unit of ___                     | size for good compression |
|-------|---------------------------------|---------------------------|
| chunk | compression                     | \>10k numbers             |
| page  | interleaving w/ wrapping format | \>1k numbers              |
| batch | decompression                   | 256 numbers (fixed)       |

### Mistakes to Avoid

You may get disappointing results from Pco if your data in a single chunk

* combines semantically different sequences, or
* contains too few numbers (see above section),
* is inherently 2D or higher.

Example: the NYC taxi dataset has `f64` columns for `fare` and
`trip_miles`.
Suppose we assign these as `fare[0...n]` and `trip_miles[0...n]` respectively, where
`n=50,000`.

* separate chunk for each column => good compression
* single chunk `fare[0], ... fare[n-1], trip_miles[0], ..., trip_miles[n-1]` => bad compression
* single chunk `fare[0], trip_miles[0], ..., fare[n-1], trip_miles[n-1]` => bad compression

## Extra

### Docs

[benchmarks: see the results](docs/benchmark_results.md)

[format specification](./docs/format.md)

[terminology](./docs/terminology.md)

[Quantile Compression: Pcodec's predecessor](./quantile-compression/README.md)

[contributing guide](./docs/CONTRIBUTING.md)

[Pcodec: Better Compression for Numerical Sequences](https://arxiv.org/abs/2502.06112) (academic paper)
* to cite:
  ```text
  @misc{pcodec,
    title={Pcodec: Better Compression for Numerical Sequences}, 
    author={Martin Loncaric and Niels Jeppesen and Ben Zinberg},
    year={2025},
    eprint={2502.06112},
    archivePrefix={arXiv},
    primaryClass={cs.IT},
    url={https://arxiv.org/abs/2502.06112}, 
  }
  ```

### Community

[join the Discord](https://discord.gg/f6eRXgMP8w)

