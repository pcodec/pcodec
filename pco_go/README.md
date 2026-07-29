# Pco for Go

Go bindings for [Pcodec](https://github.com/pcodec/pcodec), a lossless codec
for numerical sequences with a high compression ratio and moderately fast
speed.

This package wraps the reference Rust implementation through the `pco_c` C FFI,
statically linked via cgo. That means:

* compressed output is byte-for-byte identical to the Rust library's,
* every Pco format feature is supported (all modes, delta encodings, and
  format versions back to 0.0.0), and
* compression/decompression run at native Rust speed.

## Building

The package links `libcpcodec.a`, built from this repository with the Rust
toolchain:

```sh
cargo build --release -p cpcodec   # from the repo root, or `make rust` here
go test ./...
```

cgo is required (`CGO_ENABLED=1`). Cross-compiling requires a Rust static
library built for the target platform and a matching C cross-linker.

## Usage

```go
import pco "github.com/pcodec/pcodec/pco_go"

nums := []int64{1, 2, 3, 4, 5}

compressed, err := pco.Compress(nums, nil) // nil = default config (level 8)
if err != nil { ... }

decompressed, err := pco.Decompress[int64](compressed)
if err != nil { ... }
```

Supported element types: `uint8`–`uint64`, `int8`–`int64`, `float32`,
`float64`. Go has no `float16` type, so f16 data is handled as raw IEEE 754
half-precision bit patterns via `CompressFloat16Bits` / `DecompressFloat16Bits`
(`[]uint16`).

### Allocation-free variants

For hot paths, `CompressInto` and `DecompressInto` reuse caller-provided
buffers:

```go
dst := make([]byte, pco.GuaranteedFileSize[int64](len(nums)))
n, err := pco.CompressInto(dst, nums, nil)
compressed := dst[:n]

out := make([]int64, count) // count known from your own metadata or pco.Inspect
n, finished, err := pco.DecompressInto(out, compressed)
```

`Inspect` reads a file's header without decompressing, returning its number
type and element count hint (always exact for files written by this package).

All functions are safe for concurrent use from multiple goroutines; the
underlying C functions are stateless.

## Performance

On a 2.1 GHz Xeon (linux/amd64, 2^20 elements, default level):

| benchmark      | throughput | allocs/op |
|----------------|-----------:|----------:|
| Compress i64   |   301 MB/s | 1 |
| Decompress i64 |  4900 MB/s | 2 |
| Compress f64   |   164 MB/s | 1 |
| Decompress f64 |  1682 MB/s | 2 |

cgo call overhead is negligible: one C call per file operation, amortized over
the whole slice.

## Scope

This wraps Pco's **standalone** format (self-contained `.pco` files). The
lower-level **wrapped** format (embedding Pco chunks and pages inside your own
file format, with random access) is not yet exposed by the C FFI; extending
`pco_c` and this package with it is the natural next step if you need
finer-grained control.
