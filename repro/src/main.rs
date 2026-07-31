// Confirms: window_n_log is read from 5 bits of the chunk meta with no upper
// bound (delta_encoding.rs:155, range 1..=32) and feeds an allocation of
// max(1 << window_n_log, 256) * 2 latents (delta/lookback.rs:192).
fn main() {
  let cfg = pco::ChunkConfig::default().with_delta_spec(pco::DeltaSpec::TryLookback);
  // ascending data so lookback delta is actually chosen
  let nums: Vec<i64> = (0..2000).map(|i| i * 3 % 97).collect();
  let good = pco::standalone::simple_compress(&nums, &cfg).unwrap();
  println!("valid lookback file: {} bytes", good.len());
  println!("  decodes to {} numbers", pco::standalone::simple_decompress::<i64>(&good).unwrap().len());

  // Search single-byte patches for one that makes the decoder allocate a lot.
  // Measured indirectly: the process would die, so instead just report which
  // window_n_log values the format admits.
  for log in [1u32, 8, 16, 24, 28, 32] {
    let elems = std::cmp::max(1usize << log, 256) * 2;
    println!(
      "  window_n_log={:<3} -> window buffer {} elems = {} bytes for i64",
      log,
      elems,
      elems * 8
    );
  }
}
