// Microbenchmarks of pco's internals, plus end-to-end benchmarks of its public
// API on synthetic data. If you're looking for compressed sizes or benchmarks
// over real datasets, use `pcodec bench` instead.
//
// The bencmark bodies live inside the crate next to the code they measure.
// They register themselves with divan at startup; this target just runs them.
fn main() {
  pco::_bench_link();
  divan::main();
}
