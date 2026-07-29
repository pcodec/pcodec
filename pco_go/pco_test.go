package pco

import (
	"bytes"
	"errors"
	"math"
	"math/rand"
	"os"
	"path/filepath"
	"sync"
	"testing"
)

func roundTrip[T Number](t *testing.T, nums []T, config *ChunkConfig) []byte {
	t.Helper()
	compressed, err := Compress(nums, config)
	if err != nil {
		t.Fatalf("Compress: %v", err)
	}

	info, err := Inspect(compressed)
	if err != nil {
		t.Fatalf("Inspect: %v", err)
	}
	if want := numberTypeOf[T](); info.Type != want {
		t.Errorf("Inspect type = %v, want %v", info.Type, want)
	}
	if info.NHint != len(nums) {
		t.Errorf("Inspect NHint = %d, want %d", info.NHint, len(nums))
	}

	decompressed, err := Decompress[T](compressed)
	if err != nil {
		t.Fatalf("Decompress: %v", err)
	}
	if len(decompressed) != len(nums) {
		t.Fatalf("decompressed %d numbers, want %d", len(decompressed), len(nums))
	}
	for i := range nums {
		// Compare bit patterns via != on the values; for floats this treats
		// NaN != NaN, so handle them below in the float-specific test data
		// by comparing through math.Float64bits instead.
		if decompressed[i] != nums[i] && !bothNaN(nums[i], decompressed[i]) {
			t.Fatalf("mismatch at %d: got %v, want %v", i, decompressed[i], nums[i])
		}
	}
	return compressed
}

func bothNaN[T Number](a, b T) bool {
	switch x := any(a).(type) {
	case float32:
		return math.IsNaN(float64(x)) && math.IsNaN(float64(any(b).(float32)))
	case float64:
		return math.IsNaN(x) && math.IsNaN(any(b).(float64))
	default:
		return false
	}
}

func testInts[T int8 | int16 | int32 | int64 | uint8 | uint16 | uint32 | uint64](t *testing.T) {
	rng := rand.New(rand.NewSource(0))
	nums := make([]T, 3000)
	acc := T(0)
	for i := range nums {
		acc += T(rng.Intn(16))
		nums[i] = acc
	}
	var maxT, minT T
	switch any(maxT).(type) {
	case int8, int16, int32, int64:
		maxT = T(1)<<(8*int(sizeof[T]())-1) - 1
		minT = -maxT - 1
	default:
		maxT = ^T(0)
		minT = 0
	}
	nums = append(nums, minT, maxT, 0)
	roundTrip(t, nums, nil)
}

func sizeof[T Number]() uintptr {
	var z T
	switch any(z).(type) {
	case int8, uint8:
		return 1
	case int16, uint16:
		return 2
	case int32, uint32, float32:
		return 4
	default:
		return 8
	}
}

func TestRoundTripIntTypes(t *testing.T) {
	t.Run("i8", testInts[int8])
	t.Run("i16", testInts[int16])
	t.Run("i32", testInts[int32])
	t.Run("i64", testInts[int64])
	t.Run("u8", testInts[uint8])
	t.Run("u16", testInts[uint16])
	t.Run("u32", testInts[uint32])
	t.Run("u64", testInts[uint64])
}

func testFloats[T float32 | float64](t *testing.T) {
	rng := rand.New(rand.NewSource(0))
	nums := make([]T, 3000)
	for i := range nums {
		// decimal-ish values, the FloatMult-friendly case
		nums[i] = T(float64(rng.Intn(100000)) / 100.0)
	}
	nums = append(nums,
		T(math.NaN()),
		T(math.Inf(1)),
		T(math.Inf(-1)),
		T(math.Copysign(0, -1)),
		0,
		T(math.SmallestNonzeroFloat32),
	)
	roundTrip(t, nums, nil)
}

func TestRoundTripFloatTypes(t *testing.T) {
	t.Run("f32", testFloats[float32])
	t.Run("f64", testFloats[float64])
}

func TestRoundTripFloat16Bits(t *testing.T) {
	rng := rand.New(rand.NewSource(0))
	bits := make([]uint16, 2000)
	for i := range bits {
		bits[i] = uint16(rng.Intn(1 << 16))
	}
	compressed, err := CompressFloat16Bits(bits, nil)
	if err != nil {
		t.Fatalf("CompressFloat16Bits: %v", err)
	}
	info, err := Inspect(compressed)
	if err != nil {
		t.Fatalf("Inspect: %v", err)
	}
	if info.Type != TypeF16 || info.NHint != len(bits) {
		t.Errorf("Inspect = %+v, want type f16 and n %d", info, len(bits))
	}
	decompressed, err := DecompressFloat16Bits(compressed)
	if err != nil {
		t.Fatalf("DecompressFloat16Bits: %v", err)
	}
	if len(decompressed) != len(bits) {
		t.Fatalf("decompressed %d values, want %d", len(decompressed), len(bits))
	}
	for i := range bits {
		if decompressed[i] != bits[i] {
			t.Fatalf("mismatch at %d: got %#x, want %#x", i, decompressed[i], bits[i])
		}
	}
}

func TestRoundTripEmpty(t *testing.T) {
	compressed := roundTrip(t, []int64{}, nil)
	if len(compressed) == 0 {
		t.Error("empty input should still produce a valid file")
	}
}

func TestRoundTripSingle(t *testing.T) {
	roundTrip(t, []float64{3.14}, nil)
}

func TestCompressionLevels(t *testing.T) {
	nums := make([]int64, 10000)
	rng := rand.New(rand.NewSource(0))
	acc := int64(0)
	for i := range nums {
		acc += int64(rng.Intn(1000))
		nums[i] = acc
	}
	var sizes []int
	for _, level := range []int{0, 4, 8, 12} {
		compressed, err := Compress(nums, &ChunkConfig{CompressionLevel: level})
		if err != nil {
			t.Fatalf("level %d: %v", level, err)
		}
		got, err := Decompress[int64](compressed)
		if err != nil {
			t.Fatalf("level %d decompress: %v", level, err)
		}
		if len(got) != len(nums) {
			t.Fatalf("level %d: got %d numbers, want %d", level, len(got), len(nums))
		}
		sizes = append(sizes, len(compressed))
	}
	if sizes[len(sizes)-1] > sizes[0] {
		t.Errorf("level 12 (%d bytes) should not compress worse than level 0 (%d bytes)",
			sizes[len(sizes)-1], sizes[0])
	}
}

func TestCompressDeterministic(t *testing.T) {
	rng := rand.New(rand.NewSource(7))
	nums := make([]float64, 5000)
	for i := range nums {
		nums[i] = rng.NormFloat64() * 1000
	}
	a, err := Compress(nums, nil)
	if err != nil {
		t.Fatal(err)
	}
	b, err := Compress(nums, nil)
	if err != nil {
		t.Fatal(err)
	}
	if !bytes.Equal(a, b) {
		t.Error("compression of identical input should be byte-identical")
	}
}

func TestDecompressIntoPartial(t *testing.T) {
	nums := make([]int32, 1000)
	for i := range nums {
		nums[i] = int32(i * i)
	}
	compressed, err := Compress(nums, nil)
	if err != nil {
		t.Fatal(err)
	}

	small := make([]int32, 300)
	n, finished, err := DecompressInto(small, compressed)
	if err != nil {
		t.Fatalf("DecompressInto: %v", err)
	}
	if finished {
		t.Error("expected finished=false with undersized dst")
	}
	if n != len(small) {
		t.Errorf("n = %d, want %d (dst should be filled)", n, len(small))
	}
	for i := 0; i < n; i++ {
		if small[i] != nums[i] {
			t.Fatalf("mismatch at %d", i)
		}
	}

	exact := make([]int32, len(nums))
	n, finished, err = DecompressInto(exact, compressed)
	if err != nil || !finished || n != len(nums) {
		t.Fatalf("exact dst: n=%d finished=%v err=%v", n, finished, err)
	}

	oversized := make([]int32, len(nums)+123)
	n, finished, err = DecompressInto(oversized, compressed)
	if err != nil || !finished || n != len(nums) {
		t.Fatalf("oversized dst: n=%d finished=%v err=%v", n, finished, err)
	}
}

func TestCompressInto(t *testing.T) {
	nums := []uint32{1, 2, 3, 4, 5, 100, 1000}
	dst := make([]byte, GuaranteedFileSize[uint32](len(nums)))
	n, err := CompressInto(dst, nums, nil)
	if err != nil {
		t.Fatalf("CompressInto: %v", err)
	}
	got, err := Decompress[uint32](dst[:n])
	if err != nil {
		t.Fatalf("Decompress: %v", err)
	}
	if len(got) != len(nums) {
		t.Fatalf("got %d numbers, want %d", len(got), len(nums))
	}

	// undersized destination must error, not panic or overflow
	_, err = CompressInto(make([]byte, 3), nums, nil)
	if !errors.Is(err, ErrCompression) {
		t.Errorf("undersized dst: err = %v, want ErrCompression", err)
	}
}

func TestTypeMismatch(t *testing.T) {
	compressed, err := Compress([]int64{1, 2, 3}, nil)
	if err != nil {
		t.Fatal(err)
	}
	_, err = Decompress[float32](compressed)
	if !errors.Is(err, ErrInvalidType) {
		t.Errorf("err = %v, want ErrInvalidType", err)
	}
}

func TestCorruptInput(t *testing.T) {
	valid, err := Compress([]int64{1, 2, 3, 4, 5}, nil)
	if err != nil {
		t.Fatal(err)
	}
	cases := map[string][]byte{
		"garbage":   {0xde, 0xad, 0xbe, 0xef, 0x01, 0x02, 0x03},
		"empty":     {},
		"bad magic": []byte("not a pco file at all"),
		"truncated": valid[:len(valid)/2],
	}
	for name, data := range cases {
		_, err := Decompress[int64](data)
		if !errors.Is(err, ErrCorruption) && !errors.Is(err, ErrInsufficientData) {
			t.Errorf("%s: err = %v, want ErrCorruption or ErrInsufficientData", name, err)
		}
	}

	if _, err := Decompress[int64]([]byte("not a pco file at all")); !errors.Is(err, ErrCorruption) {
		t.Errorf("bad magic: err = %v, want ErrCorruption", err)
	}
}

// TestMaliciousCountHint rewrites a valid file's header to claim ~10^12
// elements. Decompress must not trust that hint with a huge allocation; it
// must still decode the actual contents.
func TestMaliciousCountHint(t *testing.T) {
	valid, err := Compress([]int64{1, 2, 3}, nil)
	if err != nil {
		t.Fatal(err)
	}
	// Header layout: "pco!", standalone version byte, uniform type byte, then
	// the count-hint varint: 6 bits (bit count - 1), that many bits of value,
	// zero-padded to a byte boundary. For n_hint=3 the varint is 6 bits of 1
	// then 2 bits of 3, exactly the single byte 0xC1.
	if len(valid) < 7 || valid[6] != 0xC1 {
		t.Fatalf("unexpected header layout; varint byte = %#x", valid[6])
	}
	// Splice in a 41-bit varint claiming n_hint = 2^40 (~10^12 elements):
	// 6 bits of 40, then 41 bits with only the top bit set, padded to 6 bytes.
	crafted := append([]byte{}, valid[:6]...)
	crafted = append(crafted, 40, 0, 0, 0, 0, 0x40)
	crafted = append(crafted, valid[7:]...)

	info, err := Inspect(crafted)
	if err != nil {
		t.Fatalf("Inspect: %v", err)
	}
	if info.NHint != 1<<40 {
		t.Fatalf("crafted NHint = %d, want %d", info.NHint, uint64(1)<<40)
	}

	got, err := Decompress[int64](crafted)
	if err != nil {
		t.Fatalf("Decompress: %v", err)
	}
	if len(got) != 3 || got[0] != 1 || got[2] != 3 {
		t.Fatalf("got %v, want [1 2 3]", got)
	}
}

// Ported from pco_java StandaloneTest.testIllegalArgument.
func TestInvalidCompressionLevel(t *testing.T) {
	nums := []int16{1, 2, 3}
	for _, level := range []int{13, -1} {
		_, err := Compress(nums, &ChunkConfig{CompressionLevel: level})
		if !errors.Is(err, ErrInvalidArgument) {
			t.Errorf("level %d: err = %v, want ErrInvalidArgument", level, err)
		}
	}
}

// Ported from pco_python test_round_trip_simple_decompress, which uses a
// paging spec to split the data across multiple chunks.
func TestMultiChunkPaging(t *testing.T) {
	rng := rand.New(rand.NewSource(0))
	nums := make([]float32, 900)
	for i := range nums {
		nums[i] = float32(rng.Float64() * 1000)
	}
	config := &ChunkConfig{CompressionLevel: DefaultCompressionLevel, MaxPageN: 300}
	compressed := roundTrip(t, nums, config)

	// partial decompression ending mid-file, past a chunk boundary
	partial := make([]float32, 500)
	n, finished, err := DecompressInto(partial, compressed)
	if err != nil || finished || n != len(partial) {
		t.Fatalf("partial across chunks: n=%d finished=%v err=%v", n, finished, err)
	}
	for i := range partial {
		if partial[i] != nums[i] {
			t.Fatalf("mismatch at %d", i)
		}
	}
}

// Ported from pco_python test_simple_decompress_into_errors.
func TestDecompressIntoWrongType(t *testing.T) {
	nums := make([]float32, 100)
	for i := range nums {
		nums[i] = float32(i)
	}
	compressed, err := Compress(nums, nil)
	if err != nil {
		t.Fatal(err)
	}
	_, _, err = DecompressInto(make([]float64, 100), compressed)
	if !errors.Is(err, ErrCorruption) {
		t.Errorf("err = %v, want ErrCorruption", err)
	}
}

// Ported from pco_python test_simple_decompress_errors: byte-level
// manipulations of a real v0.4.5 file with a uniform type. In that file,
// byte 5 is the uniform number type and byte 8 is the first chunk's number
// type.
func TestUniformTypeAssetErrors(t *testing.T) {
	orig, err := os.ReadFile("../pco/assets/v0_4_5_uniform_type.pco")
	if err != nil {
		t.Skipf("asset not found: %v", err)
	}
	if info, err := Inspect(orig); err != nil || info.Type != TypeU32 {
		t.Fatalf("expected a u32 uniform-type file, got %+v err=%v", info, err)
	}

	if _, err := Decompress[uint32](orig[:8]); !errors.Is(err, ErrInsufficientData) {
		t.Errorf("truncated: err = %v, want ErrInsufficientData", err)
	}

	patched := append([]byte{}, orig...)
	patched[8] = 99 // chunk type disagrees with uniform type
	if _, err := Decompress[uint32](patched); !errors.Is(err, ErrCorruption) {
		t.Errorf("mismatched chunk type: err = %v, want ErrCorruption", err)
	}

	patched[8] = 0 // termination byte: a valid file with no chunks
	got, err := Decompress[uint32](patched)
	if err != nil || len(got) != 0 {
		t.Errorf("chunkless file: got %d numbers, err = %v; want empty", len(got), err)
	}

	patched[5] = 0 // additionally drop the uniform type declaration
	info, err := Inspect(patched)
	if err != nil || info.Type != TypeInvalid {
		t.Errorf("no uniform type: Inspect = %+v, err = %v; want TypeInvalid", info, err)
	}
	got, err = Decompress[uint32](patched)
	if err != nil || len(got) != 0 {
		t.Errorf("chunkless untyped file: got %d numbers, err = %v; want empty", len(got), err)
	}
}

// Data shaped for each of Pco's non-classic compression modes. The C FFI
// does not yet expose ModeSpec, so modes can't be forced; these shapes make
// the automatic detection likely to engage, and in any case must round-trip.
func TestModeShapedData(t *testing.T) {
	rng := rand.New(rand.NewSource(0))

	t.Run("int mult", func(t *testing.T) {
		nums := make([]int64, 2000)
		for i := range nums {
			nums[i] = 777*int64(rng.Intn(100000)) + int64(rng.Intn(3))
		}
		roundTrip(t, nums, nil)
	})
	t.Run("float mult", func(t *testing.T) {
		nums := make([]float64, 2000)
		for i := range nums {
			nums[i] = 0.01 * float64(rng.Intn(1000000))
		}
		roundTrip(t, nums, nil)
	})
	t.Run("float quant", func(t *testing.T) {
		nums := make([]float64, 2000)
		for i := range nums {
			nums[i] = float64(float32(rng.NormFloat64()))
		}
		roundTrip(t, nums, nil)
	})
	t.Run("dict", func(t *testing.T) {
		distinct := []float64{3.14, 2.71, 1.41, 1.61, 0.577}
		nums := make([]float64, 2000)
		for i := range nums {
			nums[i] = distinct[rng.Intn(len(distinct))]
		}
		roundTrip(t, nums, nil)
	})
}

func TestConcurrentUse(t *testing.T) {
	var wg sync.WaitGroup
	for g := 0; g < 8; g++ {
		wg.Add(1)
		go func(seed int64) {
			defer wg.Done()
			rng := rand.New(rand.NewSource(seed))
			nums := make([]int64, 20000)
			acc := int64(0)
			for i := range nums {
				acc += int64(rng.Intn(100))
				nums[i] = acc
			}
			for iter := 0; iter < 5; iter++ {
				compressed, err := Compress(nums, nil)
				if err != nil {
					t.Errorf("goroutine %d: %v", seed, err)
					return
				}
				got, err := Decompress[int64](compressed)
				if err != nil || len(got) != len(nums) {
					t.Errorf("goroutine %d: n=%d err=%v", seed, len(got), err)
					return
				}
			}
		}(int64(g))
	}
	wg.Wait()
}

// TestCompatibilityAssets decodes the historical compressed files checked in
// under pco/assets, which cover every format version back to 0.0.0. Old files
// predate the uniform-type header field, so the element type is discovered by
// trying each type in turn.
func TestCompatibilityAssets(t *testing.T) {
	paths, err := filepath.Glob("../pco/assets/*.pco")
	if err != nil || len(paths) == 0 {
		t.Skipf("no assets found (err=%v)", err)
	}
	for _, path := range paths {
		t.Run(filepath.Base(path), func(t *testing.T) {
			data, err := os.ReadFile(path)
			if err != nil {
				t.Fatal(err)
			}
			info, err := Inspect(data)
			if err != nil {
				t.Fatalf("Inspect: %v", err)
			}
			n, typ, ok := decodeAnyType(data)
			if !ok {
				t.Fatal("no number type could decode this file")
			}
			t.Logf("type=%v n=%d (header: type=%v nHint=%d)", typ, n, info.Type, info.NHint)
			if info.Type != TypeInvalid && info.Type != typ {
				t.Errorf("decoded as %v but header declares %v", typ, info.Type)
			}
			if info.NHint != 0 && info.NHint != n {
				t.Errorf("decoded %d numbers but header hints %d", n, info.NHint)
			}
		})
	}
}

func decodeAnyType(data []byte) (n int, typ NumberType, ok bool) {
	if v, err := Decompress[uint8](data); err == nil {
		return len(v), TypeU8, true
	}
	if v, err := Decompress[uint16](data); err == nil {
		return len(v), TypeU16, true
	}
	if v, err := Decompress[uint32](data); err == nil {
		return len(v), TypeU32, true
	}
	if v, err := Decompress[uint64](data); err == nil {
		return len(v), TypeU64, true
	}
	if v, err := Decompress[int8](data); err == nil {
		return len(v), TypeI8, true
	}
	if v, err := Decompress[int16](data); err == nil {
		return len(v), TypeI16, true
	}
	if v, err := Decompress[int32](data); err == nil {
		return len(v), TypeI32, true
	}
	if v, err := Decompress[int64](data); err == nil {
		return len(v), TypeI64, true
	}
	if v, err := Decompress[float32](data); err == nil {
		return len(v), TypeF32, true
	}
	if v, err := Decompress[float64](data); err == nil {
		return len(v), TypeF64, true
	}
	if v, err := DecompressFloat16Bits(data); err == nil {
		return len(v), TypeF16, true
	}
	return 0, TypeInvalid, false
}
