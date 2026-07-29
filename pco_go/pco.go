// Package pco provides Go bindings for Pcodec (Pco), a lossless codec for
// numerical sequences with a high compression ratio and moderately fast
// speed.
//
// This package wraps the reference Rust implementation through the pco_c C
// FFI (linked statically via cgo), so its compressed output is byte-for-byte
// identical to the Rust library's and it supports every Pco format feature.
//
// It exposes the standalone .pco file format via a simple API: Compress turns
// a slice of numbers into a self-contained compressed file, and Decompress
// reverses it. All functions are safe for concurrent use; the underlying C
// functions are stateless and operate only on caller-provided buffers.
//
// Building requires the Rust static library. From the repository root:
//
//	cargo build --release -p cpcodec
//
// or run `make rust` in this directory. See README.md for details.
package pco

/*
#cgo CFLAGS: -I${SRCDIR}/../pco_c/include
#cgo linux LDFLAGS: ${SRCDIR}/../target/release/libcpcodec.a -lm -ldl -lpthread
#cgo darwin LDFLAGS: ${SRCDIR}/../target/release/libcpcodec.a -lm
#include <stdlib.h>
#include "cpcodec_generated.h"
*/
import "C"

import (
	"errors"
	"fmt"
	"unsafe"
)

// Number is the set of Go element types Pco can compress directly.
//
// Go has no float16 type; use the Float16Bits functions to work with f16 data
// as raw IEEE 754 half-precision bit patterns in []uint16.
type Number interface {
	uint8 | uint16 | uint32 | uint64 |
		int8 | int16 | int32 | int64 |
		float32 | float64
}

// NumberType identifies a Pco number type on the wire. The values match the
// 1-byte representations in the Pco format spec (docs/format.md).
type NumberType uint8

const (
	TypeInvalid NumberType = 0
	TypeU32     NumberType = 1
	TypeU64     NumberType = 2
	TypeI32     NumberType = 3
	TypeI64     NumberType = 4
	TypeF32     NumberType = 5
	TypeF64     NumberType = 6
	TypeU16     NumberType = 7
	TypeI16     NumberType = 8
	TypeF16     NumberType = 9
	TypeU8      NumberType = 10
	TypeI8      NumberType = 11
)

func (t NumberType) String() string {
	switch t {
	case TypeU8:
		return "u8"
	case TypeU16:
		return "u16"
	case TypeU32:
		return "u32"
	case TypeU64:
		return "u64"
	case TypeI8:
		return "i8"
	case TypeI16:
		return "i16"
	case TypeI32:
		return "i32"
	case TypeI64:
		return "i64"
	case TypeF16:
		return "f16"
	case TypeF32:
		return "f32"
	case TypeF64:
		return "f64"
	default:
		return fmt.Sprintf("invalid(%d)", uint8(t))
	}
}

// elemSize returns the size of one element of t in bytes.
func (t NumberType) elemSize() int {
	switch t {
	case TypeU8, TypeI8:
		return 1
	case TypeU16, TypeI16, TypeF16:
		return 2
	case TypeU32, TypeI32, TypeF32:
		return 4
	case TypeU64, TypeI64, TypeF64:
		return 8
	default:
		return 0
	}
}

var (
	// ErrInvalidType indicates an unsupported number type byte, or a
	// decompression request whose element type disagrees with the type
	// recorded in the compressed file.
	ErrInvalidType = errors.New("pco: invalid or mismatched number type")
	// ErrInvalidArgument indicates invalid parameters, e.g. an out-of-range
	// compression level.
	ErrInvalidArgument = errors.New("pco: invalid argument")
	// ErrCorruption indicates the compressed input is inconsistent or
	// violates the pco format.
	ErrCorruption = errors.New("pco: corrupt data")
	// ErrInsufficientData indicates the compressed input ended before
	// decompression finished, i.e. it is truncated.
	ErrInsufficientData = errors.New("pco: insufficient data (truncated input)")
	// ErrCompression indicates the underlying compressor failed for a reason
	// other than ErrInvalidArgument, e.g. because the destination buffer was
	// smaller than GuaranteedFileSize requires.
	ErrCompression = errors.New("pco: compression error")
	// ErrDecompression indicates the underlying decompressor failed for a
	// reason other than ErrCorruption or ErrInsufficientData.
	ErrDecompression = errors.New("pco: decompression error")
)

func errFromC(code C.enum_PcoError) error {
	switch code {
	case C.PcoSuccess:
		return nil
	case C.PcoInvalidType:
		return ErrInvalidType
	case C.PcoInvalidArgumentError:
		return ErrInvalidArgument
	case C.PcoCorruptionError:
		return ErrCorruption
	case C.PcoInsufficientDataError:
		return ErrInsufficientData
	case C.PcoCompressionError:
		return ErrCompression
	case C.PcoDecompressionError:
		return ErrDecompression
	default:
		return fmt.Errorf("pco: unknown error code %d", int(code))
	}
}

// ChunkConfig configures compression.
//
// The zero value is not the default configuration; use nil or DefaultConfig()
// where a *ChunkConfig is accepted to get defaults.
type ChunkConfig struct {
	// CompressionLevel ranges from 0 (fastest) to 12 (best ratio).
	// Levels beyond 8 offer little marginal ratio at real speed cost.
	CompressionLevel int
	// MaxPageN is the maximum number of elements per page.
	// 0 means the library default (2^18 = 262144).
	MaxPageN int
}

// DefaultCompressionLevel matches pco::DEFAULT_COMPRESSION_LEVEL.
const DefaultCompressionLevel = 8

// DefaultConfig returns the default compression configuration.
func DefaultConfig() *ChunkConfig {
	return &ChunkConfig{CompressionLevel: DefaultCompressionLevel}
}

func (c *ChunkConfig) toC() *C.struct_PcoChunkConfig {
	if c == nil {
		return nil
	}
	return &C.struct_PcoChunkConfig{
		compression_level: C.uint(c.CompressionLevel),
		max_page_n:        C.size_t(c.MaxPageN),
	}
}

// numberTypeOf maps a Go element type to its Pco wire type.
func numberTypeOf[T Number]() NumberType {
	var z T
	switch any(z).(type) {
	case uint8:
		return TypeU8
	case uint16:
		return TypeU16
	case uint32:
		return TypeU32
	case uint64:
		return TypeU64
	case int8:
		return TypeI8
	case int16:
		return TypeI16
	case int32:
		return TypeI32
	case int64:
		return TypeI64
	case float32:
		return TypeF32
	case float64:
		return TypeF64
	default:
		panic("unreachable: type outside Number constraint")
	}
}

// placeholder gives cgo a valid non-nil pointer for empty slices; the C side
// never dereferences a pointer whose accompanying length is 0, but Rust slice
// construction requires the pointer to be non-null.
var placeholder byte

func basePtr[E any](s []E) unsafe.Pointer {
	if len(s) == 0 {
		return unsafe.Pointer(&placeholder)
	}
	return unsafe.Pointer(unsafe.SliceData(s))
}

// GuaranteedFileSize returns the maximum possible compressed size, in bytes,
// of a standalone file holding n elements of type T. A destination buffer of
// this size always suffices for CompressInto.
func GuaranteedFileSize[T Number](n int) int {
	return guaranteedFileSize(n, numberTypeOf[T]())
}

// GuaranteedFileSizeFloat16Bits is GuaranteedFileSize for f16 data.
func GuaranteedFileSizeFloat16Bits(n int) int {
	return guaranteedFileSize(n, TypeF16)
}

func guaranteedFileSize(n int, dtype NumberType) int {
	if n < 0 {
		return 0
	}
	return int(C.pco_standalone_guarantee_file_size(C.size_t(n), C.uchar(dtype)))
}

// Compress compresses nums into a standalone .pco file. A nil config selects
// defaults (compression level 8).
func Compress[T Number](nums []T, config *ChunkConfig) ([]byte, error) {
	return compress(basePtr(nums), len(nums), numberTypeOf[T](), config)
}

// CompressFloat16Bits compresses IEEE 754 half-precision values, given as raw
// bit patterns, into a standalone .pco file.
func CompressFloat16Bits(bits []uint16, config *ChunkConfig) ([]byte, error) {
	return compress(basePtr(bits), len(bits), TypeF16, config)
}

func compress(nums unsafe.Pointer, n int, dtype NumberType, config *ChunkConfig) ([]byte, error) {
	dst := make([]byte, guaranteedFileSize(n, dtype))
	written, err := compressInto(dst, nums, n, dtype, config)
	if err != nil {
		return nil, err
	}
	// The guarantee buffer is sized for incompressible data; copy to an
	// exact-size allocation so callers don't retain the oversized backing
	// array.
	out := make([]byte, written)
	copy(out, dst[:written])
	return out, nil
}

// CompressInto compresses nums into dst and returns the number of bytes
// written. dst must be at least GuaranteedFileSize[T](len(nums)) bytes;
// otherwise compression may fail with ErrCompression.
//
// This variant exists to let callers reuse output buffers; Compress is
// otherwise equivalent.
func CompressInto[T Number](dst []byte, nums []T, config *ChunkConfig) (int, error) {
	return compressInto(dst, basePtr(nums), len(nums), numberTypeOf[T](), config)
}

func compressInto(dst []byte, nums unsafe.Pointer, n int, dtype NumberType, config *ChunkConfig) (int, error) {
	var written C.size_t
	code := C.pco_standalone_simple_compress_into(
		nums,
		C.size_t(n),
		C.uchar(dtype),
		config.toC(),
		basePtr(dst),
		C.size_t(len(dst)),
		&written,
	)
	if err := errFromC(code); err != nil {
		return 0, err
	}
	return int(written), nil
}

// FileInfo describes a standalone .pco file's header.
type FileInfo struct {
	// Type is the file's uniform number type, or TypeInvalid if the file does
	// not declare one (possible for files from other writers; files written
	// by this package always declare it).
	Type NumberType
	// NHint is the total number of elements in the file if recorded at
	// compression time, or 0 if unknown. Files written by this package always
	// record an exact count.
	NHint int
}

// Inspect reads a standalone .pco file's header without decompressing it.
func Inspect(compressed []byte) (FileInfo, error) {
	var dtype C.uchar
	var nHint C.size_t
	code := C.pco_standalone_file_info(
		basePtr(compressed),
		C.size_t(len(compressed)),
		&dtype,
		&nHint,
	)
	if err := errFromC(code); err != nil {
		return FileInfo{}, err
	}
	return FileInfo{Type: NumberType(dtype), NHint: int(nHint)}, nil
}

// Decompress decompresses a standalone .pco file of element type T.
//
// If the file declares a uniform number type that is not T, Decompress
// returns ErrInvalidType.
func Decompress[T Number](compressed []byte) ([]T, error) {
	n, dst, err := decompressAlloc[T](compressed, numberTypeOf[T]())
	if err != nil {
		return nil, err
	}
	return dst[:n], nil
}

// DecompressFloat16Bits decompresses a standalone .pco file of f16 values
// into their raw IEEE 754 half-precision bit patterns.
func DecompressFloat16Bits(compressed []byte) ([]uint16, error) {
	n, dst, err := decompressAlloc[uint16](compressed, TypeF16)
	if err != nil {
		return nil, err
	}
	return dst[:n], nil
}

func decompressAlloc[E Number](compressed []byte, dtype NumberType) (int, []E, error) {
	info, err := Inspect(compressed)
	if err != nil {
		return 0, nil, err
	}
	if info.Type != TypeInvalid && info.Type != dtype {
		return 0, nil, fmt.Errorf("%w: file holds %s, requested %s",
			ErrInvalidType, info.Type, dtype)
	}

	// NHint is exact for files written by pco's simple compressors, so the
	// grow loop below is normally never entered. It is still untrusted input:
	// a tiny crafted file could claim an enormous count, so bound the initial
	// allocation by a multiple of the compressed size. Legitimate files with
	// extreme compression ratios are necessarily small, so re-decompressing
	// them for a few doubling passes is cheap.
	capacity := info.NHint
	if capacity == 0 {
		capacity = max(256, len(compressed)/dtype.elemSize())
	}
	capacity = min(capacity, max(64*1024, 256*len(compressed)))
	for {
		dst := make([]E, capacity)
		n, finished, err := decompressInto(dst, compressed, dtype)
		if err != nil {
			return 0, nil, err
		}
		if finished {
			if n <= capacity/2 {
				// The clamp or a lying hint left dst mostly empty; don't make
				// the caller retain the oversized backing array.
				exact := make([]E, n)
				copy(exact, dst[:n])
				return n, exact, nil
			}
			return n, dst, nil
		}
		capacity = 2 * capacity
	}
}

// DecompressInto decompresses a standalone .pco file of element type T into
// dst, filling as many elements as fit. It returns the number of elements
// written and whether the whole file was decompressed. An undersized dst is
// not an error: n elements are written and finished is false.
//
// This variant exists to let callers reuse buffers when the element count is
// known (e.g. recorded in their own metadata, or via Inspect); Decompress is
// otherwise equivalent.
func DecompressInto[T Number](dst []T, compressed []byte) (n int, finished bool, err error) {
	return decompressInto(dst, compressed, numberTypeOf[T]())
}

// DecompressIntoFloat16Bits is DecompressInto for f16 data as raw bits.
func DecompressIntoFloat16Bits(dst []uint16, compressed []byte) (n int, finished bool, err error) {
	return decompressInto(dst, compressed, TypeF16)
}

func decompressInto[E Number](dst []E, compressed []byte, dtype NumberType) (int, bool, error) {
	var written C.size_t
	var finished C.uchar
	code := C.pco_standalone_simple_decompress_partial_into(
		basePtr(compressed),
		C.size_t(len(compressed)),
		C.uchar(dtype),
		basePtr(dst),
		C.size_t(len(dst)),
		&written,
		&finished,
	)
	if err := errFromC(code); err != nil {
		return 0, false, err
	}
	return int(written), finished != 0, nil
}
