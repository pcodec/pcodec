# Findings

What the libFuzzer targets in `fuzz/fuzz_targets/` turned up. The fixes and
regression tests for all of these live in a separate PR; this file is kept as
the record of how each one is reached, so the targets can be re-pointed at them
later. Line numbers refer to the tree before the fixes.

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

## 4. `mult * base` overflows in `int_mult::join_latents`

`pco/src/mode/int_mult.rs:48` multiplies two latents read from the wire with a
plain `*`, while the adjacent addition on the same line is already
`wrapping_add`. Overflow panics under overflow-checks; in a stock release build
it wraps, which is what the neighbouring `wrapping_add` shows was intended.

Fix direction: `wrapping_mul` (needs adding to the `LatentPriv` trait).

## 5. Unbounded window buffer from `window_n_log` (confirmed Kani lead)

`pco/src/metadata/delta_encoding.rs:155` reads
`window_n_log = 1 + read_bitlen(5)`, i.e. **1..=32**, and validates only
`state_n_log <= window_n_log`. `delta/lookback.rs:192` then allocates
`max(1 << window_n_log, 256) * 2` latents.

The encoder never emits more than `LOOKBACK_MAX_WINDOW_N_LOG = 15`
(`delta/mod.rs:16`) -- the decoder accepts more than the encoder can produce.

From a valid **165-byte** lookback file:

| window_n_log | window buffer (i64) |
| --- | --- |
| 15 (encoder max) | 512 KiB |
| 24 | 256 MiB |
| 28 | 4 GiB  <- what the fuzzer hit |
| 32 (format max) | 64 GiB |

Affects release builds. Fix direction: reject `window_n_log` above the
encoder's max on read.

## Build-mode matrix

`cargo fuzz` defaults to opt-level 3 **plus** debug-assertions and
overflow-checks -- not a stock release build. Each finding was re-checked in
both, via `repro/` and by replaying artifacts under `cargo fuzz run -O`:

| # | stock release | debug-assertions / overflow-checks |
| --- | --- | --- |
| 1 n_hint | **abort / panic** | abort / panic |
| 2 FloatQuant k | clean `InvalidArgument` | **panic** |
| 3 FloatQuant debug_assert | check absent, arithmetic wraps | **panic** |
| 4 int_mult multiply | wraps (intended) | **panic** |
| 5 window_n_log | **multi-GB allocation** | multi-GB allocation |

So 1 and 5 are the ones that hit shipping builds; 2, 3 and 4 abort any consumer
that builds with overflow checks on, and in release leave arithmetic running on
data that violates a stated invariant.

## Post-fix status

All five fixed locally (see git log). Full pco suite still green (128 + 5).
Then, with no crash found:

* `decompress_corrupt`, opt3 + debug-assertions: **1 029 368 runs / 301 s**
* `decompress_corrupt`, stock release (`-O`): **801 490 runs / 301 s**
