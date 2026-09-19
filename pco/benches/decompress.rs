// Microbenchmarks of pco's decompression internals.
//
// The benchmark bodies live inside the crate (next to the code they measure,
// like unit tests) because most of the routines are private. They register
// themselves with divan at startup; this target just runs them.
//
// Run with: cargo bench -p pco --features bench --bench decompress
//
// Note: on x86_64, .cargo/config.toml enables bmi1/bmi2/lzcnt/avx2. Those flags
// do not apply on aarch64, so numbers are not comparable across architectures.

fn main() {
  // Required: without a reference into pco, the linker discards the object
  // files holding the benches and divan finds nothing to run.
  pco::_bench_link();
  divan::main();
}
