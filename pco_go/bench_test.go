package pco

import (
	"math/rand"
	"testing"
)

const benchN = 1 << 20

func benchNumsI64() []int64 {
	rng := rand.New(rand.NewSource(0))
	nums := make([]int64, benchN)
	acc := int64(0)
	for i := range nums {
		acc += int64(rng.Intn(1000))
		nums[i] = acc
	}
	return nums
}

func benchNumsF64() []float64 {
	rng := rand.New(rand.NewSource(0))
	nums := make([]float64, benchN)
	for i := range nums {
		nums[i] = float64(rng.Intn(1000000)) / 100.0
	}
	return nums
}

func BenchmarkCompressI64(b *testing.B) {
	nums := benchNumsI64()
	dst := make([]byte, GuaranteedFileSize[int64](len(nums)))
	b.SetBytes(int64(8 * len(nums)))
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		if _, err := CompressInto(dst, nums, nil); err != nil {
			b.Fatal(err)
		}
	}
}

func BenchmarkDecompressI64(b *testing.B) {
	nums := benchNumsI64()
	compressed, err := Compress(nums, nil)
	if err != nil {
		b.Fatal(err)
	}
	dst := make([]int64, len(nums))
	b.SetBytes(int64(8 * len(nums)))
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		if _, _, err := DecompressInto(dst, compressed); err != nil {
			b.Fatal(err)
		}
	}
}

func BenchmarkCompressF64(b *testing.B) {
	nums := benchNumsF64()
	dst := make([]byte, GuaranteedFileSize[float64](len(nums)))
	b.SetBytes(int64(8 * len(nums)))
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		if _, err := CompressInto(dst, nums, nil); err != nil {
			b.Fatal(err)
		}
	}
}

func BenchmarkDecompressF64(b *testing.B) {
	nums := benchNumsF64()
	compressed, err := Compress(nums, nil)
	if err != nil {
		b.Fatal(err)
	}
	dst := make([]float64, len(nums))
	b.SetBytes(int64(8 * len(nums)))
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		if _, _, err := DecompressInto(dst, compressed); err != nil {
			b.Fatal(err)
		}
	}
}
