# Pco Format Spec

This document aims to describe the Pco format exactly.

All values encoded are unsigned integers.
All bit packing (and thus integer encoding) is done in a little-endian fashion.
Bit packing a component is completed by filling the rest of the byte with 0s.

Let `dtype_size` be the number type's number of bits.
A "raw" value for a number is a `dtype_size` value that maps to the number
via [its `from_unsigned` function](#Modes).

## Version Compatibility

Define "compatibility line" to be a non-API-breaking sequence of SemVer
versions, e.g 0.4.x or 1.x.y.
Pco's compatibility guarantee is:

Each library version will be able to decompress any data compressed by
* earlier or equal library versions (in the sense of SemVer precedence), and
* later library versions in the same compatibility line, unless opt-in features
  are added to the API and opted into by the user during compression.

Note that we allow:
* New library versions may support decompressing data that was previously
  considered corrupt.
* Data produced by new compatibility lines may be considered corrupt by old
  compatibility lines.
* Within a compatibility line, opt-in compressor features may be added that
  produce data that was previously considered corrupt.

## Wrapped Format Components

The wrapped format consists of 3 components: header, chunk metadata, and data
pages.
Wrapping formats may encode these components any place they wish.
Pco is designed to have one header per file, possibly multiple chunks per
header, and possibly multiple pages per chunk.

[Plate notation](https://en.wikipedia.org/wiki/Plate_notation) for chunk
metadata component:

<img alt="Pco wrapped chunk meta plate notation" src="../images/wrapped_chunk_meta_plate.svg" width="500px"/>

Plate notation for page component:

<img alt="Pco wrapped page plate notation" src="../images/wrapped_page_plate.svg" width="420px"/>

### Header

Pco's header specifies its major and (as of major version 4) minor versions,
allowing Pco to make small format changes in the future if necessary.
To explain the meaning of these: suppose a decompressor supports version `a.b`, and is reading a file with version `c.d`.
* If `a < c`, the decompressor will definitely be unable to read the file; the
  file has modifications the decompressor does not support.
* If `a >= c, b < d`, the decompressor might be able to read the file; the file
  may or may not contain additions the decompressor does not support.
* If `a >= c, b >= d`, the decompressor will definitely be able to read the
  file.


The header simply consists of
* [8 bits] the major format version
* [8 bits] the minor format version

So far, these format versions exist:

| format version | 1st reader lib version | 1st writer lib version | format modifications    | format additions              |
|----------------|------------------------|------------------------|-------------------------|-------------------------------|
| 0              | 0.0.0                  | 0.0.0                  |                         |                               |
| 1              | 0.1.0                  | 0.1.0                  |                         | IntMult mode                  |
| 2              | 0.3.0                  | 0.3.0                  |                         | FloatQuant mode, 16-bit types |
| 3              | 0.4.0                  | 0.4.0                  | delta encoding variants | Lookback delta encoding       |
| 4.0            | 0.4.8                  | -                      | minor version           |                               |
| 4.1            | 1.0.0                  | 1.0.0                  |                         | Dict mode                     |

### Chunk Metadata

It is expected that decompressors raise corruption errors if any part of
metadata is out of range.
For example, if the sum of bin weights does not equal the tANS size; or if a
bin's offset bits exceed the data type size.

Each chunk meta consists of

* [4 bits] `mode`, using this table:

  | value | mode         | primary latent | secondary latent  | `extra_mode_bits` |
  |-------|--------------|----------------|-------------------|-------------------|
  | 0     | Classic      | `T::L`         |                   | 0                 |
  | 1     | IntMult      | `T::L`         | `T::L`            | `dtype_size`      |
  | 2     | FloatMult    | `T::L`         | `T::L`            | `dtype_size`      |
  | 3     | FloatQuant   | `T::L`         | `T::L`            | 8                 |
  | 4     | Dict         | `u32`          |                   | variable*         |
  | 5-15  | \<reserved\> |                |                   |                   |

  Here, `T::L` refers to the latent type with the same number of bits as the
  number type, e.g. u64 for i64.
  `Dict` mode's payload is a 25-bit integer for the count of values in the
  dictionary, followed by those raw values.

* [`extra_mode_bits` bits] for certain modes, extra data is parsed. See the
  mode-specific formulas below for how this is used, e.g. as the `mult`, `k`, or
  `dict` values.
  The value encoded in these bits should be validated; namely, mult mode bases
  should be finite and nonzero, quant mode must have `0 < k <= MANTISSA_BITS`,
  int modes cannot apply to floats, and vice versa.
  Parsing the `dict` value is more complex than the others: it is formed by
  reading 25 bits for `dict_len`, followed by 0s until byte-aligned, followed by
  `dict_len` raw values that consitute `dict`.
* [4 bits] `delta_encoding`, using this table:

  | value | delta encoding | n latent variables | `extra_delta_bits` |
  |-------|----------------|--------------------|--------------------|
  | 0     | None           | 0                  | 0                  |
  | 1     | Consecutive    | 0                  | 4                  |
  | 2     | Lookback       | 1                  | 10                 |
  | 3     | Conv1          | 0                  | variable           |
  | 4-15  | \<reserved\>   |                    |                    |

* [`extra_delta_bits` bits]
  * for `consecutive`, this is 3 bits for `order` from 1-7, and 1 bit for
    whether the mode's secondary latent is delta encoded.
    An order of 0 is considered a corruption.
    Let `state_n = order`.
  * for `lookback`, this is 5 bits for `window_n_log - 1`, 4 for
    `state_n_log`, and 1 for whether the mode's secondary latent is delta
    encoded.
    Let `state_n = 1 << state_n_log`.
  * for `conv1`, this is 5 bits for `quantization`, 64 bits for a raw `bias`
    value (i64), 5 bits for the order (number of weights), and 32 bits per raw
    weight value (i32).
* per latent variable (ordered by delta latent variables followed by mode
  latent variables),
  * [4 bits] `ans_size_log`, the log2 of the size of its tANS table.
    This may not exceed 14.
  * [15 bits] the count of bins
  * per bin,
    * [`ans_size_log` bits] 1 less than `weight`, this bin's weight in the tANS table
    * [`dtype_size` bits] the lower bound of this bin's numerical range,
      encoded as a raw value.
    * [`log2(dtype_size) + 1` bits] the number of offset bits for this bin
      e.g. for a 64-bit data type, this will be 7 bits long.

Based on chunk metadata, 4-way interleaved tANS decoders should be initialized
using
[the simple `spread_state_tokens` algorithm from this repo](../pco/src/ans/spec.rs).

### Page

If there are `n` numbers in a page, it will consist of `ceil(n / 256)`
batches. All but the final batch will contain 256 numbers, and the final
batch will contain the rest (<= 256 numbers).

Each page consists of

* per latent variable,
  * if delta encoding is applicable, for `i in 0..state_n`,
    * [`dtype_size` bits] the `i`th delta state
  * for `i in 0..4`,
    * [`ans_size_log` bits] the `i`th interleaved tANS state index
* [0-7 bits] 0s until byte-aligned
* per batch of `k` numbers,
  * per latent variable,
    * for `i in 0..k`,
      * [tANS state `i % 4`'s bits] tANS encoded bin idx for the `i`th
        latent. Store the bin as `bin[i]`. Asymmetric Numeral System links:
        [original paper](https://arxiv.org/abs/0902.0271),
        [blog post explanation](https://graphallthethings.com/posts/streaming-ans-explained).
    * for `i in 0..k`,
      * [`bin[i].offset_bits` bits] offset for `i`th latent

## Standalone Format

The standalone format is a minimal implementation of the wrapped format.
It consists of

* [32 bits] magic header (ASCII for "pco!")
* [8 bits] standalone version
* [8 bits] either a uniform number type which all following chunks must share,
  or 0.
* [6 bits] 1 less than `n_hint_log2`
* [`n_hint_log2` bits] `n_hint`, the total count of numbers in the file, if known;
  0 otherwise
* [0-7 bits] 0s until byte-aligned
* a wrapped header
* per chunk,
  * [8 bits] the number type
  * [24 bits] 1 less than `chunk_n`, the count of numbers in the chunk
  * a wrapped chunk metadata
  * a wrapped page of `chunk_n` numbers
* [8 bits] a magic termination byte (0).

So far, these standalone versions exist:

| format version | first Rust version | format modifications |
|----------------|--------------------|----------------------|
| 0              | 0.0.0              |   |
| 1              | 0.1.0              |   |
| 2              | 0.1.1              | explicit standalone version (previously was implicit and equaled wrapped major version), added `n_hint` |
| 3              | 0.4.5              | uniform number type |

As well as these number type 1-byte representations:

| number type | byte |
|-------------|------|
| f16         | 9 |
| f32         | 5 |
| f64         | 6 |
| i16         | 8 |
| i32         | 3 |
| i64         | 4 |
| u16         | 7 |
| u32         | 1 |
| u64         | 2 |

## Processing Formulas

In order of decompression steps in a batch:

### Bin Indices and Offsets -> Latents

To produce latents, we simply do `l[i] = bin[i].lower + offset[i]`.

### Delta Encodings

Depending on `delta_encoding`, the mode latents are further decoded.
Note that the delta latent variable, if it exists, is never delta encoded
itself.

#### None

No additional processing is applied.

##### Consecutive

Latents are decoded by taking a cumulative sum repeatedly.
The delta state is interpreted as delta moments, which are used to initialize
each cumulative sum, and get modified for the next batch.

For instance, with 2nd order delta encoding, the delta moments `[1, 2]`
and the deltas `[0, 10, 0]` would decode to the latents `[1, 3, 5, 17, 29]`.

#### Lookback

Letting `lookback` be the delta latent variable.
Mode latents are decoded via `l[i] += l[i - lookback[i]]`.

The decompressor should error if any lookback exceeds the window.

### Conv1

Supposing the latents are k-bit, conv1 arithmetic is mainly done in 2k-bit
signed values to avoid overflow.
Latents are decoded in order via
`l[i] += ((bias + sum[weight[j] * l[i - order + j]]) >> quantization) as L`.
The bit shift here is arithmetic, not logical.

### Modes

Based on the mode, latents are joined into the finalized numbers.
Let `l0` and `l1` be the primary and secondary latents respectively.

| mode       | decoding formula                                                       |
|------------|------------------------------------------------------------------------|
| Classic    | `from_latent_ordered(l0)`                                              |
| Dict       | `from_latent_ordered(dict[l0])`                                        |
| IntMult    | `from_latent_ordered(l0 * mult + l1)`                                  |
| FloatMult  | `int_float_from_latent(l0) * mult + (l1 + MID) ULPs`                   |
| FloatQuant | `from_latent_ordered((l0 << k) + (l0 << k >= MID ? l1 : 2^k - 1 - l1)` |

Here ULP refers to [unit in the last place](https://en.wikipedia.org/wiki/Unit_in_the_last_place),
`MID` is the middle value for the latent type (e.g. 2^31 for `u32`), and `dict` is a dictionary of unique values, stored in as the Dict mode metadata payload.
Each number type has an order-preserving bijection to an unsigned latent type
known as `from_latent_ordered` and `to_latent_ordered`.
For instance, floats have their first bit toggled, and the rest of their bits
toggled if the float was originally negative:

```rust
fn from_unsigned(unsigned: u32) -> f32 {
  if unsigned & (1 << 31) > 0 {
    // positive float
    f32::from_bits(unsigned ^ (1 << 31))
  } else {
    // negative float
    f32::from_bits(!unsigned)
  }
}
```

Signed integers have an order-preserving wrapping addition and wrapping
conversion like so:

```rust
fn from_unsigned(unsigned: u32) -> i32 {
  i32::MIN.wrapping_add(unsigned as i32)
}
```
