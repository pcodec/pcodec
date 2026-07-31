# Findings

Setup: Kani proofs (`pco/src/kani_proofs.rs`, `pco/src/delta/lookback.rs`) and
libFuzzer targets (`fuzz/fuzz_targets/`). Repro binary: `repro/src/main.rs`.

## 1. Unbounded `Vec::with_capacity` from the header's `n_hint` (worst)

`n_hint` is a varint in the standalone file header
(`pco/src/standalone/decompressor.rs:113`). It is a *hint* -- nothing ties it to
the amount of data present -- yet `simple_decompress` passes it straight to
`Vec::with_capacity` (`pco/src/standalone/decompressor.rs:251`).

* `n_hint = 2^40`, **26-byte** file -> attempts an 8.8 TB allocation -> process
  **abort** (uncatchable, not a `PcoResult`).
* `n_hint = usize::MAX`, **34-byte** file -> `capacity overflow` **panic**.
* Reached from 13 bytes of unstructured input by `decompress_arbitrary`
  (172 GB allocation attempt).

Affects release builds. The files are produced by pco's own public API
(`FileCompressor::with_n_hint`); no bit-twiddling required. pco's own test
suite writes `with_n_hint(usize::MAX)` (`pco/src/standalone/guarantee.rs:56`),
so this is a legal value on the write side that the read side cannot survive.

Fix direction: treat `n_hint` as advisory -- clamp against remaining input
length, or just grow the vec on demand.

## 2. `ModeSpec::TryFloatQuant(k)` panics for `k >= L::BITS` before validation

`pco/src/data_types/float.rs:117` calls `float_quant::split_latents(nums, k)`
*before* the mode-validity check that would return `InvalidArgument`.
`split_latents` does `(F::L::ONE << k) - F::L::ONE`
(`pco/src/mode/float_quant.rs:46`).

* debug / overflow-checks on: `TryFloatQuant(64)` on `f64` -> shift-overflow
  **panic**. Public API, well-formed input, no corruption.
* release: the shift is masked to `k % 64`, silently wrong, but the downstream
  validity check then returns `InvalidArgument`, so the user sees an error.
* `k = 63` returns `InvalidArgument` in both -- the boundary is exactly at
  `L::BITS`.

Fix direction: validate `k < L::BITS` before splitting latents.

## 3. `debug_assert!` used to validate untrusted wire data

`pco/src/mode/float_quant.rs:26` asserts `m >> k == 0` on latents read from the
file. A **single-byte** patch (offset 16) of a valid 125-byte FloatQuant file
panics on decompress in a debug build; in release the assert is compiled out
and `lowest_k_bits_max - m` wraps instead.

No memory unsafety either way -- it is latent arithmetic -- but a debug build of
any consumer aborts on corrupt input, and release silently computes on data
that violates a stated invariant.

Fix direction: make it a real `PcoError::corruption` check.
