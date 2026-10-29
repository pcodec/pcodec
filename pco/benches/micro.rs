// Microbenchmarks of pco's internals. If you're looking for end-to-end
// benchmarks or compressed sizes, use `pcodec bench` instead.
//
// The bencmark bodies live inside the crate next to the code they measure.
// They register themselves with divan at startup; this target just runs them.
fn main() {
  pco::_bench_link();
  divan::main();
}
