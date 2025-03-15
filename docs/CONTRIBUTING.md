We're thrilled to get feedback and code contributions on pcodec!

**If you think you've found a bug or other issue**, please file a
Github issue.

**If you have a feature request or general question**, it's best to
[join our Discord](https://discord.gg/f6eRXgMP8w) for a quick response. If
you're opposed to creating a Discord account, Github issues also work.

# Code Contribution

* If you're thinking of implementing something, it's best to chat with us
  about it first. That way we can vet your idea and make sure your efforts won't
  be in vain.
* Before making a PR, make sure to
  * Test your code; `cargo test` and `cargo clippy`.
  * Format it; `cargo fmt`.
  * [Run the benchmarks](benchmark_results.md).
    This verifies compression and
    decompression works for each synthetic dataset, which occasionally catches
    strange cases the tests miss.
    Also, if your change might affect performance, compare relevant runtimes to
    the benchmarks on the main branch.

Looking for ideas on what to contribute? Grep through the repo for concrete
TODOs, or look at our
[project ideas](https://github.com/pcodec/pcodec/wiki/pcodec-project-ideas)
for harder, underspecified problems.

# Deploying Packages

This is entirely managed by @mwlon right now, but just for reference:

## Rust / Crates.io

`pco` and `pco_cli` are manually deployed with `cargo publish` from a local
clone of the repo.

## Python / PyPi

`pco_python` is packaged by
[a Github workflow](../.github/workflows/python_ci.yml)
whenever the release name contains "Python". This runs a lot of maturin builds,
each of which produces a dynamic library for a targets (in the sense of OS /
hardware tuples). Each such package is automatically published to PyPi.
